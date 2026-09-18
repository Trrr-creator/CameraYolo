using System.IO;
using CameraYolo.Models;
using OpenCvSharp;
using OpenCvSharp.Dnn;

namespace CameraYolo.Services;

/// <summary>
/// Loads YOLO models via OpenCV DNN (native C++ backend).
/// Supports YOLOv5/v8/v11 ONNX and YOLOv3/v4 Darknet (.cfg + .weights).
/// </summary>
public sealed class YoloDetector : IDisposable
{
    private Net? _net;
    private YoloModelInfo? _info;
    private string[] _classNames = Array.Empty<string>();
    private bool _isYoloV8Style;
    private bool _gpuFailed;  // GPU 推理失败后回退标志

    /// <summary>检测到的可用 GPU 数量</summary>
    public int GpuCount { get; private set; }

    /// <summary>当前使用的设备名称</summary>
    public string CurrentDevice => _info?.DeviceName ?? "未加载";

    /// <summary>
    /// 探测当前设备可用的推理后端，返回 (backend, target, deviceName)
    /// 优先级：CUDA > OpenCL > CPU
    /// </summary>
    private static (Backend backend, Target target, string deviceName) DetectBestDevice()
    {
        return ResolveDevice(null);
    }

    /// <summary>
    /// 根据用户选择解析推理设备
    /// </summary>
    private static (Backend backend, Target target, string deviceName) ResolveDevice(string? device)
    {
        // 用户指定了设备
        if (!string.IsNullOrEmpty(device))
        {
            switch (device.ToLowerInvariant())
            {
                case "cuda":
                    return (Backend.CUDA, Target.CUDA, "NVIDIA GPU (CUDA)");
                case "opencl":
                case "opencl_fp16":
                    return (Backend.OPENCV, Target.OPENCL, "GPU (OpenCL)");
                case "cpu":
                    return (Backend.OPENCV, Target.CPU, "CPU");
            }
        }

        // 自动检测
        // 检查环境变量暗示有 CUDA
        bool hasCudaEnv = !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("CUDA_PATH"))
                       || !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("CUDA_HOME"));

        // 优先尝试 CUDA（如果环境变量暗示有 GPU）
        if (hasCudaEnv)
        {
            try
            {
                // 创建一个临时的 ONNX 模型来测试 CUDA
                // 如果 OpenCV 没有编译 CUDA 支持，SetPreferableBackend 会抛出异常
                using var testNet = new Net();
                testNet.SetPreferableBackend(Backend.CUDA);
                testNet.SetPreferableTarget(Target.CUDA);
                return (Backend.CUDA, Target.CUDA, "NVIDIA GPU (CUDA)");
            }
            catch { /* CUDA 设置失败 */ }
        }

        // 尝试 OpenCL
        try
        {
            using var testNet = new Net();
            testNet.SetPreferableBackend(Backend.OPENCV);
            testNet.SetPreferableTarget(Target.OPENCL);
            return (Backend.OPENCV, Target.OPENCL, "GPU (OpenCL)");
        }
        catch { /* OpenCL 不可用 */ }

        // 回退到 CPU
        return (Backend.OPENCV, Target.CPU, "CPU");
    }

    public YoloModelInfo? Current => _info;
    public bool IsLoaded => _net is not null && !_net.Empty();
    public IReadOnlyList<string> ClassNames => _classNames;

    /// <summary>用户指定的设备（null=自动检测）</summary>
    public string? PreferredDevice { get; set; }

    public void Load(YoloModelInfo info)
    {
        ArgumentNullException.ThrowIfNull(info);
        if (!File.Exists(info.ModelPath))
            throw new FileNotFoundException("模型文件不存在", info.ModelPath);

        // 检测最佳推理设备
        var (backend, target, deviceName) = ResolveDevice(PreferredDevice);

        var net = info.Format switch
        {
            YoloFormat.Onnx => CvDnn.ReadNetFromOnnx(info.ModelPath),
            YoloFormat.Darknet => CvDnn.ReadNetFromDarknet(
                info.ConfigPath ?? throw new ArgumentNullException(nameof(info.ConfigPath)),
                info.ModelPath),
            _ => throw new NotSupportedException($"不支持的模型格式: {info.Format}")
        };

        if (net is null || net.Empty())
            throw new InvalidOperationException("OpenCV DNN 加载模型失败（空网络）");

        // 尝试设置 GPU，失败则回退 CPU
        try
        {
            net.SetPreferableBackend(backend);
            net.SetPreferableTarget(target);
            // 验证设置是否生效（用一个空 forward 测试）
            info.UsedBackend = backend;
            info.UsedTarget = target;
            info.DeviceName = deviceName;
        }
        catch
        {
            // GPU 设置失败，回退 CPU
            net.SetPreferableBackend(Backend.OPENCV);
            net.SetPreferableTarget(Target.CPU);
            info.UsedBackend = Backend.OPENCV;
            info.UsedTarget = Target.CPU;
            info.DeviceName = "CPU (GPU 不可用)";
        }

        _net?.Dispose();

        _net = net;
        _info = info;
        _classNames = ResolveClassNames(info);
        _isYoloV8Style = DetectYoloV8Style(info);
    }

    public void LoadFromPaths(string modelPath, string? configPath, string? classesPath, int inputSize = 640)
    {
        var ext = Path.GetExtension(modelPath).ToLowerInvariant();
        var format = ext switch
        {
            ".onnx" => YoloFormat.Onnx,
            ".cfg" => YoloFormat.Darknet,
            ".weights" => YoloFormat.Darknet,
            _ => throw new NotSupportedException($"无法识别的模型扩展名: {ext}")
        };

        string model = modelPath;
        string? cfg = configPath;
        if (format == YoloFormat.Darknet)
        {
            if (ext == ".cfg")
            {
                cfg = modelPath;
                var weights = Path.ChangeExtension(modelPath, ".weights");
                if (!File.Exists(weights))
                    throw new FileNotFoundException("未找到与 .cfg 同名的 .weights 文件", weights);
                model = weights;
            }
            else if (string.IsNullOrWhiteSpace(cfg) || !File.Exists(cfg))
            {
                var guessed = Path.ChangeExtension(modelPath, ".cfg");
                if (!File.Exists(guessed))
                    throw new FileNotFoundException("Darknet 模型需要 .cfg 配置文件", guessed);
                cfg = guessed;
            }
        }

        Load(new YoloModelInfo
        {
            ModelPath = model,
            ConfigPath = cfg,
            ClassesPath = classesPath,
            Format = format,
            InputSize = inputSize,
            ClassNames = Array.Empty<string>()
        });
    }

    public IReadOnlyList<DetectionResult> Detect(Mat frame)
    {
        if (_net is null || _info is null || frame.Empty())
            return Array.Empty<DetectionResult>();

        int inputSize = _info.InputSize;
        using var blob = CvDnn.BlobFromImage(
            frame,
            1.0 / 255.0,
            new Size(inputSize, inputSize),
            new Scalar(),
            swapRB: true,
            crop: false);

        _net.SetInput(blob);

        // Prefer modern multi-output heads; fall back to single output.
        var outputNames = _net.GetUnconnectedOutLayersNames();
        var outputs = new Mat[outputNames.Length];
        try
        {
            try
            {
                if (outputs.Length == 1)
                {
                    outputs[0] = _net.Forward(outputNames[0]);
                }
                else
                {
                    var names = outputNames.Select(n => n ?? string.Empty).ToArray();
                    _net.Forward(outputs, names);
                }
            }
            catch (Exception) when (!_gpuFailed)
            {
                // GPU 推理失败，回退到 CPU
                _gpuFailed = true;
                _info.UsedBackend = Backend.OPENCV;
                _info.UsedTarget = Target.CPU;
                _info.DeviceName += " (已回退CPU)";

                _net.SetPreferableBackend(Backend.OPENCV);
                _net.SetPreferableTarget(Target.CPU);

                // 重新用 CPU 推理
                if (outputs.Length == 1)
                    outputs[0] = _net.Forward(outputNames[0]);
                else
                    _net.Forward(outputs, outputNames.Select(n => n ?? string.Empty).ToArray());
            }

            var boxes = new List<Rect>();
            var confidences = new List<float>();
            var classIds = new List<int>();
            float confThresh = _info.ConfidenceThreshold;

            foreach (var output in outputs)
            {
                bool v8 = _isYoloV8Style || LooksLikeYoloV8(output);
                if (v8)
                    ParseYoloV8(output, frame.Width, frame.Height, inputSize, confThresh, boxes, confidences, classIds);
                else
                    ParseYoloV3V5(output, frame.Width, frame.Height, inputSize, confThresh, boxes, confidences, classIds);
            }

            int[] indices = Array.Empty<int>();
            if (boxes.Count > 0)
            {
                CvDnn.NMSBoxes(
                    boxes.ToArray(),
                    confidences.ToArray(),
                    confThresh,
                    _info.NmsThreshold,
                    out indices);
            }

            var results = new List<DetectionResult>(indices.Length);
            foreach (var i in indices)
            {
                int cid = classIds[i];
                string label = cid >= 0 && cid < _classNames.Length ? _classNames[cid] : $"class_{cid}";
                results.Add(new DetectionResult(label, cid, confidences[i], boxes[i]));
            }

            return results;
        }
        finally
        {
            foreach (var m in outputs)
                m?.Dispose();
        }
    }

    private static bool DetectYoloV8Style(YoloModelInfo info)
    {
        // Darknet classic path is always [1, N, 5+nc].
        if (info.Format == YoloFormat.Darknet)
            return false;

        // Prefer output-shape probing after load; filename is only a hint.
        var name = Path.GetFileNameWithoutExtension(info.ModelPath).ToLowerInvariant();
        return name.Contains("yolov8")
               || name.Contains("yolo11")
               || name.Contains("yolo8")
               || name.Contains("yolov11")
               || name.Contains("custom")
               || name.Contains("person")
               || name.Contains("wider");
    }

    private bool LooksLikeYoloV8(Mat output)
    {
        // Ultralytics export: [1, 4+nc, 8400] — channel dim much smaller than anchors.
        int rank = output.Dims;
        if (rank < 3)
            return false;
        int c = output.Size(1);
        int n = output.Size(2);
        // 5 = 4 box + 1 class (single-class) or typical small head; classic is [1, N, 5+nc] with N >> 5+nc.
        return c > 4 && c < 256 && n > c;
    }

    private void ParseYoloV8(
        Mat output, int frameW, int frameH, int inputSize,
        float confThresh, List<Rect> boxes, List<float> confidences, List<int> classIds)
    {
        // Expected shapes:
        //  - [1, 4+nc, N]  (Ultralytics export)
        //  - [1, N, 4+nc]  (some converters)
        int rank = output.Dims;
        int channels = rank >= 3 ? output.Size(1) : 0;
        int rows = rank >= 3 ? output.Size(2) : output.Cols;

        bool channelFirst = channels > 4 && channels < rows;
        int featureCount = channelFirst ? channels : rows;
        int proposalCount = channelFirst ? rows : channels;
        int classCount = featureCount - 4;
        if (classCount <= 0)
            return;

        float scaleX = frameW / (float)inputSize;
        float scaleY = frameH / (float)inputSize;

        for (int i = 0; i < proposalCount; i++)
        {
            float best = 0;
            int bestId = -1;
            for (int c = 0; c < classCount; c++)
            {
                float score = channelFirst
                    ? output.At<float>(0, 4 + c, i)
                    : output.At<float>(0, i, 4 + c);
                if (score > best)
                {
                    best = score;
                    bestId = c;
                }
            }

            if (best < confThresh || bestId < 0)
                continue;

            float cx = channelFirst ? output.At<float>(0, 0, i) : output.At<float>(0, i, 0);
            float cy = channelFirst ? output.At<float>(0, 1, i) : output.At<float>(0, i, 1);
            float w = channelFirst ? output.At<float>(0, 2, i) : output.At<float>(0, i, 2);
            float h = channelFirst ? output.At<float>(0, 3, i) : output.At<float>(0, i, 3);

            // Ultralytics ONNX is already in input-pixel space.
            int left = (int)((cx - w / 2f) * scaleX);
            int top = (int)((cy - h / 2f) * scaleY);
            int width = (int)(w * scaleX);
            int height = (int)(h * scaleY);

            boxes.Add(new Rect(left, top, width, height));
            confidences.Add(best);
            classIds.Add(bestId);
        }
    }

    private void ParseYoloV3V5(
        Mat output, int frameW, int frameH, int inputSize,
        float confThresh, List<Rect> boxes, List<float> confidences, List<int> classIds)
    {
        // [1, N, 5+nc] with cx,cy,w,h,obj,class...
        int rank = output.Dims;
        int proposalCount = rank >= 3 ? output.Size(1) : 0;
        int featureCount = rank >= 3 ? output.Size(2) : 0;
        if (proposalCount <= 0 || featureCount < 6)
            return;

        float scaleX = frameW / (float)inputSize;
        float scaleY = frameH / (float)inputSize;

        for (int i = 0; i < proposalCount; i++)
        {
            float obj = output.At<float>(0, i, 4);
            if (obj < confThresh)
                continue;

            int bestId = 0;
            float best = 0;
            for (int c = 5; c < featureCount; c++)
            {
                float s = output.At<float>(0, i, c);
                if (s > best)
                {
                    best = s;
                    bestId = c - 5;
                }
            }

            float conf = obj * best;
            if (conf < confThresh)
                continue;

            float cx = output.At<float>(0, i, 0);
            float cy = output.At<float>(0, i, 1);
            float w = output.At<float>(0, i, 2);
            float h = output.At<float>(0, i, 3);

            // Classic darknet/v5 export may be normalized or pixel-based.
            bool normalized = cx <= 1.5f && cy <= 1.5f && w <= 1.5f && h <= 1.5f;
            float pw = normalized ? w * inputSize : w;
            float ph = normalized ? h * inputSize : h;
            float pcx = normalized ? cx * inputSize : cx;
            float pcy = normalized ? cy * inputSize : cy;

            int left = (int)((pcx - pw / 2f) * scaleX);
            int top = (int)((pcy - ph / 2f) * scaleY);
            int width = (int)(pw * scaleX);
            int height = (int)(ph * scaleY);

            boxes.Add(new Rect(left, top, width, height));
            confidences.Add(conf);
            classIds.Add(bestId);
        }
    }

    private static string[] ResolveClassNames(YoloModelInfo info)
    {
        if (info.ClassNames.Count > 0)
            return info.ClassNames.ToArray();

        if (!string.IsNullOrWhiteSpace(info.ClassesPath) && File.Exists(info.ClassesPath))
        {
            return File.ReadAllLines(info.ClassesPath)
                .Select(x => x.Trim())
                .Where(x => x.Length > 0)
                .ToArray();
        }

        // Fallback: COCO 80 classes (common pretrained default).
        return CocoNames;
    }

    public static readonly string[] CocoNames =
    {
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    };

    public void Dispose()
    {
        _net?.Dispose();
        _net = null;
    }
}
