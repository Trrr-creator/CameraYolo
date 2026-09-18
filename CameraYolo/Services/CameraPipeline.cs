using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using CameraYolo.Models;
using OpenCvSharp;
using Rect = OpenCvSharp.Rect;

namespace CameraYolo.Services;

public sealed class CameraPipeline : IDisposable
{
    private VideoCapture? _capture;
    private Thread? _loopThread;
    private CancellationTokenSource? _cts;
    private readonly object _sync = new();
    private YoloDetector? _detector;
    private volatile bool _detectEnabled = true;
    private int _cameraIndex;

    public int CaptureWidth { get; set; } = 640;
    public int CaptureHeight { get; set; } = 360;
    public int DetectEveryNFrames { get; set; } = 2;
    public bool DropFramesWhenBusy { get; set; } = true;

    private WriteableBitmap? _bitmap;
    private int _bmpW, _bmpH;
    private int _uiBusy;
    private int _inferBusy;
    private volatile bool _acceptFrames;
    private volatile int _frameEpoch;

    /// <summary>采集会话序号；停止/重启递增，用于丢弃迟到帧。</summary>
    public int FrameEpoch => _frameEpoch;

    /// <summary>仅在运行中为 true；停止后 UI 忽略一切帧回调。</summary>
    public bool IsAcceptingFrames => _acceptFrames;

    public event Action? PreviewCleared;
    public event Action<BitmapSource>? FrameReady;
    public event Action<IReadOnlyList<DetectionResult>, double>? DetectionReady;
    public event Action<string>? StatusChanged;

    public bool IsRunning { get; private set; }
    public double LastFps { get; private set; }
    public int LastDetectionCount { get; private set; }

    public void SetDetector(YoloDetector? detector)
    {
        lock (_sync) { _detector = detector; }
    }

    public void SetDetectEnabled(bool enabled) => _detectEnabled = enabled;

    public void Start(int cameraIndex)
    {
        if (IsRunning)
            return;

        _cameraIndex = cameraIndex;
        _frameEpoch++;
        _acceptFrames = true;
        _cts = new CancellationTokenSource();
        var token = _cts.Token;
        int epoch = _frameEpoch;
        _loopThread = new Thread(() => CaptureLoop(token, epoch))
        {
            IsBackground = true,
            Name = "CameraLoop",
            Priority = ThreadPriority.AboveNormal
        };
        IsRunning = true;
        _loopThread.Start();
        StatusChanged?.Invoke($"已启动摄像头 #{cameraIndex} @ {CaptureWidth}x{CaptureHeight}");
    }

    public void Stop()
    {
        // 先禁止收帧，再关线程，最后清 UI —— 避免清空后又被旧帧写回
        _acceptFrames = false;
        _frameEpoch++;

        if (!IsRunning && _loopThread is null && _capture is null)
        {
            ClearPreviewState();
            return;
        }

        IsRunning = false;
        try { _cts?.Cancel(); } catch { /* ignore */ }
        _loopThread?.Join(1500);
        _loopThread = null;
        try { _cts?.Dispose(); } catch { /* ignore */ }
        _cts = null;

        lock (_sync)
        {
            _capture?.Dispose();
            _capture = null;
        }

        ClearPreviewState();
        StatusChanged?.Invoke("摄像头已停止（预览已清空）");
    }

    private void ClearPreviewState()
    {
        _bitmap = null;
        _bmpW = 0;
        _bmpH = 0;
        _uiBusy = 0;
        _inferBusy = 0;
        LastFps = 0;
        LastDetectionCount = 0;
        PreviewCleared?.Invoke();
    }

    private void CaptureLoop(CancellationToken token, int epoch)
    {
        VideoCapture? capture = null;
        try
        {
            capture = new VideoCapture(_cameraIndex, VideoCaptureAPIs.ANY);
            if (!capture.IsOpened())
            {
                capture.Dispose();
                capture = new VideoCapture(_cameraIndex);
            }

            if (!capture.IsOpened())
            {
                StatusChanged?.Invoke($"无法打开摄像头 #{_cameraIndex}");
                IsRunning = false;
                _acceptFrames = false;
                return;
            }

            capture.Set(VideoCaptureProperties.FrameWidth, CaptureWidth);
            capture.Set(VideoCaptureProperties.FrameHeight, CaptureHeight);
            try { capture.Set(VideoCaptureProperties.BufferSize, 1); } catch { /* ignore */ }

            lock (_sync) { _capture = capture; }

            using var frame = new Mat();
            var fpsWatch = System.Diagnostics.Stopwatch.StartNew();
            int frames = 0;
            int frameIndex = 0;
            IReadOnlyList<DetectionResult> lastDets = Array.Empty<DetectionResult>();
            double lastInferMs = 0;

            bool Alive() => !token.IsCancellationRequested && _acceptFrames && epoch == _frameEpoch;

            while (Alive())
            {
                if (!capture.Read(frame) || frame.Empty())
                {
                    Thread.Sleep(5);
                    continue;
                }

                frameIndex++;
                var dets = lastDets;
                var inferMs = lastInferMs;

                int n = Math.Max(1, DetectEveryNFrames);
                bool doInfer = _detectEnabled && (n <= 1 || frameIndex % n == 0);

                if (doInfer && Interlocked.CompareExchange(ref _inferBusy, 1, 0) == 0)
                {
                    YoloDetector? det;
                    lock (_sync) { det = _detector; }

                    if (det is { IsLoaded: true })
                    {
                        var sw = System.Diagnostics.Stopwatch.StartNew();
                        try
                        {
                            dets = det.Detect(frame);
                            lastDets = dets;
                        }
                        catch (Exception ex)
                        {
                            StatusChanged?.Invoke($"推理异常: {ex.Message}");
                        }

                        inferMs = sw.Elapsed.TotalMilliseconds;
                        lastInferMs = inferMs;
                    }

                    Interlocked.Exchange(ref _inferBusy, 0);
                }
                else if (!_detectEnabled)
                {
                    dets = Array.Empty<DetectionResult>();
                    lastDets = dets;
                }

                if (!Alive())
                    break;

                DrawDetections(frame, dets);
                LastDetectionCount = dets.Count;

                if (DropFramesWhenBusy && Interlocked.CompareExchange(ref _uiBusy, 1, 0) != 0)
                {
                    frames++;
                    if (fpsWatch.Elapsed.TotalSeconds >= 1.0)
                    {
                        LastFps = frames / fpsWatch.Elapsed.TotalSeconds;
                        frames = 0;
                        fpsWatch.Restart();
                        StatusChanged?.Invoke($"FPS {LastFps:F1} | 目标 {LastDetectionCount} | 推理 {lastInferMs:F0} ms");
                    }

                    continue;
                }

                if (!Alive())
                {
                    Interlocked.Exchange(ref _uiBusy, 0);
                    break;
                }

                BitmapSource? bitmap = null;
                try
                {
                    bitmap = MatToWriteableBitmap(frame);
                    bitmap.Freeze();
                }
                catch
                {
                    Interlocked.Exchange(ref _uiBusy, 0);
                }

                if (!Alive())
                {
                    Interlocked.Exchange(ref _uiBusy, 0);
                    break;
                }

                if (bitmap is not null)
                    FrameReady?.Invoke(bitmap);

                DetectionReady?.Invoke(dets, inferMs);

                frames++;
                if (fpsWatch.Elapsed.TotalSeconds >= 1.0)
                {
                    LastFps = frames / fpsWatch.Elapsed.TotalSeconds;
                    frames = 0;
                    fpsWatch.Restart();
                    StatusChanged?.Invoke($"FPS {LastFps:F1} | 目标 {LastDetectionCount} | 推理 {inferMs:F0} ms");
                }
            }
        }
        catch (Exception ex)
        {
            StatusChanged?.Invoke($"摄像头线程异常: {ex.Message}");
        }
        finally
        {
            _acceptFrames = false;
            lock (_sync)
            {
                if (ReferenceEquals(_capture, capture))
                    _capture = null;
            }

            capture?.Dispose();
            IsRunning = false;
        }
    }

    public void NotifyUiFrameDone() => Interlocked.Exchange(ref _uiBusy, 0);

    private WriteableBitmap MatToWriteableBitmap(Mat mat)
    {
        if (mat.Empty())
            throw new ArgumentException("Empty Mat", nameof(mat));

        int w = mat.Width;
        int h = mat.Height;
        int srcStep = (int)mat.Step();

        if (_bitmap is null || _bmpW != w || _bmpH != h)
        {
            _bitmap = new WriteableBitmap(w, h, 96, 96, PixelFormats.Bgr24, null);
            _bmpW = w;
            _bmpH = h;
        }

        var wb = _bitmap;
        wb.Lock();
        try
        {
            unsafe
            {
                var src = (byte*)mat.Data;
                var dst = (byte*)wb.BackBuffer;
                int dstStride = wb.BackBufferStride;
                int copy = Math.Min(srcStep, dstStride);
                for (int y = 0; y < h; y++)
                    Buffer.MemoryCopy(src + y * srcStep, dst + y * dstStride, dstStride, copy);
            }

            wb.AddDirtyRect(new Int32Rect(0, 0, w, h));
        }
        finally
        {
            wb.Unlock();
        }

        return new WriteableBitmap(wb);
    }

    private static void DrawDetections(Mat frame, IReadOnlyList<DetectionResult> dets)
    {
        foreach (var d in dets)
        {
            var color = ColorForClass(d.ClassId);
            Cv2.Rectangle(frame, d.Box, color, 2, LineTypes.Link8);
            var text = $"{d.Label} {d.Confidence * 100:F0}%";
            int baseline;
            var textSize = Cv2.GetTextSize(text, HersheyFonts.HersheySimplex, 0.5, 1, out baseline);
            var textOrg = new OpenCvSharp.Point(
                Math.Max(0, d.Box.X),
                Math.Max(textSize.Height + 4, d.Box.Y - 4));
            Cv2.Rectangle(
                frame,
                new Rect(textOrg.X, textOrg.Y - textSize.Height - 4, textSize.Width + 6, textSize.Height + 8),
                color,
                -1);
            Cv2.PutText(frame, text, textOrg, HersheyFonts.HersheySimplex, 0.5, Scalar.White, 1, LineTypes.Link8);
        }
    }

    private static Scalar ColorForClass(int classId)
    {
        var palette = new[]
        {
            new Scalar(0, 220, 255),
            new Scalar(80, 200, 120),
            new Scalar(255, 120, 80),
            new Scalar(200, 80, 255),
            new Scalar(80, 160, 255),
            new Scalar(255, 200, 60)
        };
        return palette[(classId % palette.Length + palette.Length) % palette.Length];
    }

    public void Dispose() => Stop();
}
