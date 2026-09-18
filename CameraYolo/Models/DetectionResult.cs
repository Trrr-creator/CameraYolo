namespace CameraYolo.Models;

public sealed record DetectionResult(
    string Label,
    int ClassId,
    float Confidence,
    OpenCvSharp.Rect Box);
