using OpenCvSharp.Dnn;

namespace CameraYolo.Models;

public enum YoloFormat
{
    Onnx,
    Darknet
}

public sealed class YoloModelInfo
{
    public required string ModelPath { get; init; }
    public string? ConfigPath { get; init; }
    public string? ClassesPath { get; init; }
    public YoloFormat Format { get; init; }
    public int InputSize { get; init; } = 640;
    public float ConfidenceThreshold { get; set; } = 0.45f;
    public float NmsThreshold { get; set; } = 0.45f;
    public IReadOnlyList<string> ClassNames { get; init; } = Array.Empty<string>();

    /// <summary>实际使用的推理后端</summary>
    public Backend UsedBackend { get; set; } = Backend.OPENCV;

    /// <summary>实际使用的推理设备</summary>
    public Target UsedTarget { get; set; } = Target.CPU;

    /// <summary>设备描述（用于 UI 显示）</summary>
    public string DeviceName { get; set; } = "CPU";
}
