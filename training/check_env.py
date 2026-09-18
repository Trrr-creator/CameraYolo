import sys
print("py", sys.version)
print("exe", sys.executable)
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch missing", e)
try:
    import ultralytics
    print("ultra", ultralytics.__version__)
except Exception as e:
    print("ultra missing", e)
