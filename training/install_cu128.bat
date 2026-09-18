@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
"D:\Anaconda\envs\pytorch\python.exe" -m pip uninstall -y torch torchvision
if errorlevel 1 echo UNINSTALL_FAIL
"D:\Anaconda\envs\pytorch\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 echo INSTALL_FAIL
"D:\Anaconda\envs\pytorch\python.exe" -c "import torch; print(torch.__version__, torch.cuda.is_available()); import torch as t; x=t.randn(8,8,device='cuda'); print((x@x).sum().item())"
