using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using CameraYolo.Models;
using CameraYolo.Services;
using Microsoft.Win32;

namespace CameraYolo;

public partial class MainWindow : Window
{
    private readonly CameraPipeline _pipeline = new();
    private readonly YoloDetector _detector = new();
    private bool _modelLoaded;
    private int _resultUpdateCounter;

    public MainWindow()
    {
        InitializeComponent();
        EnumerateCameras();
        EnumerateDevices();

        _pipeline.PreviewCleared += OnPreviewCleared;
        _pipeline.FrameReady += OnFrameReady;
        _pipeline.DetectionReady += OnDetectionReady;
        _pipeline.StatusChanged += s => Dispatcher.BeginInvoke(() =>
        {
            if (!_pipeline.IsRunning)
            {
                StatusText.Text = s;
                FpsText.Text = "未启动";
                return;
            }

            StatusText.Text = s;
            FpsText.Text = $"FPS {_pipeline.LastFps:F1} | 目标 {_pipeline.LastDetectionCount}";
        }, DispatcherPriority.Background);

        Closed += (_, _) =>
        {
            _pipeline.Dispose();
            _detector.Dispose();
        };

        Loaded += (_, _) => TryAutoLoadDefaultModel();
    }

    private void OnPreviewCleared()
    {
        Dispatcher.BeginInvoke(() =>
        {
            PreviewImage.Source = null;
            ResultList.Items.Clear();
            PlaceholderText.Visibility = Visibility.Visible;
            PlaceholderText.Text = _modelLoaded
                ? "已停止\n\n预览已清空，点击「启动」重新检测"
                : "摄像头已停止\n\n请加载模型后启动";
            FpsText.Text = "未启动";
        }, DispatcherPriority.Normal);

        // 再清一次，冲掉 Dispatcher 队列里可能残留的旧帧写入
        Dispatcher.BeginInvoke(() =>
        {
            if (!_pipeline.IsAcceptingFrames)
            {
                PreviewImage.Source = null;
                PlaceholderText.Visibility = Visibility.Visible;
            }
        }, DispatcherPriority.Background);

        Dispatcher.BeginInvoke(() =>
        {
            if (!_pipeline.IsAcceptingFrames)
            {
                PreviewImage.Source = null;
                GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: true);
                GC.WaitForPendingFinalizers();
                GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: true);
            }
        }, DispatcherPriority.ApplicationIdle);
    }

    private void TryAutoLoadDefaultModel()
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "models", "widerperson_person.onnx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "models", "widerperson_person.onnx"),
            @"D:\OpenCV\develop\1\models\widerperson_person.onnx"
        };

        foreach (var path in candidates)
        {
            var full = Path.GetFullPath(path);
            if (!File.Exists(full))
                continue;

            try
            {
                var dir = Path.GetDirectoryName(full)!;
                string? classes = Path.Combine(dir, "classes.txt");
                if (!File.Exists(classes))
                    classes = null;

                _detector.LoadFromPaths(full, null, classes, inputSize: 640);
                _pipeline.SetDetector(_detector);
                _modelLoaded = true;

                var info = _detector.Current!;
                ModelInfoText.Text =
                    $"模型：{Path.GetFileName(info.ModelPath)}\n" +
                    $"类别：{string.Join(", ", _detector.ClassNames)}\n" +
                    $"输入：{info.InputSize}×{info.InputSize}\n" +
                    $"采集：{_pipeline.CaptureWidth}×{_pipeline.CaptureHeight}\n" +
                    $"推理间隔：每 {_pipeline.DetectEveryNFrames} 帧\n" +
                    $"推理设备：{info.DeviceName}";
                StatusText.Text = "模型已就绪，点「启动」";
                PlaceholderText.Text = "模型已就绪\n\n点击「启动」开始摄像头检测";
                return;
            }
            catch (Exception ex)
            {
                StatusText.Text = $"自动加载失败: {ex.Message}";
            }
        }

        StatusText.Text = "未找到默认模型，请手动「加载模型」";
    }

    private void EnumerateCameras()
    {
        CameraCombo.Items.Clear();
        for (int i = 0; i < 4; i++)
            CameraCombo.Items.Add(new ComboBoxItem { Content = $"摄像头 #{i}", Tag = i });
        CameraCombo.SelectedIndex = 0;
    }

    private void EnumerateDevices()
    {
        DeviceCombo.Items.Clear();
        DeviceCombo.Items.Add(new ComboBoxItem { Content = "自动检测", Tag = "auto" });
        DeviceCombo.Items.Add(new ComboBoxItem { Content = "CPU", Tag = "cpu" });
        DeviceCombo.Items.Add(new ComboBoxItem { Content = "NVIDIA GPU (CUDA)", Tag = "cuda" });
        DeviceCombo.Items.Add(new ComboBoxItem { Content = "GPU (OpenCL)", Tag = "opencl" });
        DeviceCombo.SelectedIndex = 0;
    }

    private void DeviceCombo_Changed(object sender, SelectionChangedEventArgs e)
    {
        if (!IsLoaded || DeviceCombo.SelectedItem is not ComboBoxItem item)
            return;

        var tag = item.Tag as string;
        _detector.PreferredDevice = tag == "auto" ? null : tag;

        // 如果模型已加载，切换设备需要重新加载
        if (_modelLoaded)
        {
            StatusText.Text = $"已切换到 {item.Content}，下次加载模型时生效";
        }
    }

    private void LoadModel_Click(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog
        {
            Title = "选择 YOLO 模型",
            Filter = "YOLO 模型 (*.onnx;*.weights;*.cfg)|*.onnx;*.weights;*.cfg|ONNX (*.onnx)|*.onnx|Darknet (*.weights)|*.weights|所有文件|*.*"
        };

        if (dlg.ShowDialog() != true)
            return;

        try
        {
            string? classesPath = null;
            var modelDir = Path.GetDirectoryName(dlg.FileName);
            if (!string.IsNullOrEmpty(modelDir))
            {
                foreach (var candidate in new[] { "classes.txt", "coco.names", "labels.txt" })
                {
                    var p = Path.Combine(modelDir, candidate);
                    if (File.Exists(p))
                    {
                        classesPath = p;
                        break;
                    }
                }
            }

            string? cfg = null;
            var ext = Path.GetExtension(dlg.FileName).ToLowerInvariant();
            if (ext == ".weights")
            {
                var guess = Path.ChangeExtension(dlg.FileName, ".cfg");
                if (File.Exists(guess))
                {
                    cfg = guess;
                }
                else
                {
                    var cfgDlg = new OpenFileDialog
                    {
                        Title = "选择 Darknet .cfg",
                        Filter = "Darknet cfg (*.cfg)|*.cfg"
                    };
                    if (cfgDlg.ShowDialog() == true)
                        cfg = cfgDlg.FileName;
                }
            }

            if (classesPath is null)
            {
                var clsDlg = new OpenFileDialog
                {
                    Title = "选择类别文件（可取消，使用 COCO 默认）",
                    Filter = "类别文件 (*.txt;*.names)|*.txt;*.names|所有文件|*.*"
                };
                if (clsDlg.ShowDialog() == true)
                    classesPath = clsDlg.FileName;
            }

            _detector.LoadFromPaths(dlg.FileName, cfg, classesPath, inputSize: 640);
            _pipeline.SetDetector(_detector);
            _modelLoaded = true;

            var info = _detector.Current!;
            ModelInfoText.Text =
                $"模型：{Path.GetFileName(info.ModelPath)}\n" +
                $"类别：{string.Join(", ", _detector.ClassNames)}\n" +
                $"输入：{info.InputSize}×{info.InputSize}\n" +
                $"推理设备：{info.DeviceName}";

            StatusText.Text = "模型加载成功，可以启动摄像头";
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, $"加载失败：{ex.Message}", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    private void StartBtn_Click(object sender, RoutedEventArgs e)
    {
        if (!_modelLoaded)
        {
            var r = MessageBox.Show(this,
                "尚未加载 YOLO 模型。\n仍要启动摄像头预览（无检测）吗？",
                "提示", MessageBoxButton.YesNo, MessageBoxImage.Question);
            if (r != MessageBoxResult.Yes)
                return;
            _pipeline.SetDetector(null);
        }

        int index = 0;
        if (CameraCombo.SelectedItem is ComboBoxItem { Tag: int tag })
            index = tag;

        PlaceholderText.Visibility = Visibility.Collapsed;
        PreviewImage.Source = null;
        ResultList.Items.Clear();

        // 帧率优化参数（可按机器调整）
        _pipeline.CaptureWidth = 640;
        _pipeline.CaptureHeight = 360;
        _pipeline.DetectEveryNFrames = 2;

        _pipeline.Start(index);
        StartBtn.IsEnabled = false;
        StopBtn.IsEnabled = true;
    }

    private void StopBtn_Click(object sender, RoutedEventArgs e)
    {
        // 先停管线（禁止再收帧），再同步清空图像
        _pipeline.Stop();
        PreviewImage.Source = null;
        ResultList.Items.Clear();
        PlaceholderText.Visibility = Visibility.Visible;
        PlaceholderText.Text = _modelLoaded
            ? "已停止\n\n预览已清空，点击「启动」重新检测"
            : "摄像头已停止\n\n请加载模型后启动";
        StartBtn.IsEnabled = true;
        StopBtn.IsEnabled = false;
        StatusText.Text = "已停止，预览已清空";
        FpsText.Text = "未启动";
    }

    private void DetectCheck_Changed(object sender, RoutedEventArgs e)
    {
        _pipeline.SetDetectEnabled(DetectCheck.IsChecked == true);
    }

    private void ConfSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        if (!IsLoaded)
            return;
        ConfText.Text = ConfSlider.Value.ToString("F2");
        if (_detector.Current is not null)
            _detector.Current.ConfidenceThreshold = (float)ConfSlider.Value;
    }

    private void OnFrameReady(BitmapSource bmp)
    {
        int epoch = _pipeline.FrameEpoch;
        Dispatcher.BeginInvoke(() =>
        {
            // 停止后 / 旧会话的迟到帧一律丢弃，避免“占位+旧图”同时出现
            if (!_pipeline.IsAcceptingFrames || epoch != _pipeline.FrameEpoch)
            {
                _pipeline.NotifyUiFrameDone();
                return;
            }

            PreviewImage.Source = bmp;
            PlaceholderText.Visibility = Visibility.Collapsed;
            _pipeline.NotifyUiFrameDone();
        }, DispatcherPriority.Render);
    }

    private void OnDetectionReady(IReadOnlyList<DetectionResult> dets, double inferMs)
    {
        if (!_pipeline.IsAcceptingFrames)
            return;

        _resultUpdateCounter++;
        if (_resultUpdateCounter % 3 != 0 && dets.Count == 0)
            return;

        int epoch = _pipeline.FrameEpoch;
        Dispatcher.BeginInvoke(() =>
        {
            if (!_pipeline.IsAcceptingFrames || epoch != _pipeline.FrameEpoch)
                return;

            ResultList.Items.Clear();
            foreach (var d in dets.OrderByDescending(x => x.Confidence).Take(20))
                ResultList.Items.Add($"{d.Label} {d.Confidence:P0} [{d.Box.Width}×{d.Box.Height}]");

            if (dets.Count == 0 && DetectCheck.IsChecked == true && _modelLoaded)
                ResultList.Items.Add("（当前无目标）");
        }, DispatcherPriority.Background);
    }
}
