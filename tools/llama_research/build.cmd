@echo off
setlocal
if not exist "%~dp0build" mkdir "%~dp0build"
if "%~1"=="" (
  call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
) else (
  call "%~1" >nul
)
if errorlevel 1 exit /b 1
cl /nologo /std:c++17 /EHsc /MT /O2 /W4 /utf-8 /I"%~dp0vendor" "%~dp0graph_capture.cpp" /Fo:"%~dp0build\graph_capture.obj" /Fe:"%~dp0build\llama-graph-capture.exe"
exit /b %errorlevel%
