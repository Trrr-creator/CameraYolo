# Build & run CameraYolo (sets NuGet env required on this machine)
. "$PSScriptRoot\build-env.ps1"
Set-Location $PSScriptRoot
dotnet build -c Release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Start-Process ".\CameraYolo\bin\Release\net8.0-windows\CameraYolo.exe"
