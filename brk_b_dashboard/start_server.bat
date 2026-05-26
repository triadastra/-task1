@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

echo ==========================================
echo   BRK-B 数据展示中心 - 本地服务器启动器
echo ==========================================
echo.

:: 尝试多个常见路径找 python.exe
set "PYTHON="

:: 1. 尝试直接调用 python（如果已在 PATH）
python --version >nul 2>&1
if %errorlevel% == 0 (
    set "PYTHON=python"
    goto :found
)

:: 2. 尝试 py 启动器
py --version >nul 2>&1
if %errorlevel% == 0 (
    set "PYTHON=py"
    goto :found
)

:: 3. 尝试用户目录下的 Python（你之前装过的位置）
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    goto :found
)
if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" (
    set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    goto :found
)
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    goto :found
)
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    goto :found
)

:: 4. 尝试 Program Files
if exist "C:\Program Files\Python312\python.exe" (
    set "PYTHON=C:\Program Files\Python312\python.exe"
    goto :found
)
if exist "C:\Program Files\Python313\python.exe" (
    set "PYTHON=C:\Program Files\Python313\python.exe"
    goto :found
)

:: 5. 尝试 Microsoft Store 版
if exist "%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe" (
    set "PYTHON=%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe"
    goto :found
)

:: 6. 尝试 uv（如果你用 uv 管理 Python）
if exist "%USERPROFILE%\.local\bin\uv.exe" (
    echo 未找到 Python，但检测到 uv，尝试用 uv 运行...
    cd /d "%~dp0.."
    "%USERPROFILE%\.local\bin\uv.exe" run python -m http.server 8080
    if %errorlevel% == 0 goto :done
)

:: 都没找到
echo.
echo [错误] 未找到 Python！
echo.
echo 请尝试以下方法之一：
echo 1. 安装 Python 3（https://python.org）并勾选 "Add to PATH"
echo 2. 如果你知道 python.exe 的位置，可以直接运行：
echo    ^<完整路径^>\python.exe -m http.server 8080
echo 3. 安装 uv（https://docs.astral.sh/uv）后重试
echo.
pause
exit /b 1

:found
echo 找到 Python: !PYTHON!
cd /d "%~dp0.."
echo 正在启动服务器，请稍候...
echo 访问地址: http://localhost:8080/brk_b_dashboard/
echo 按 Ctrl+C 停止服务器
echo.
"!PYTHON!" -m http.server 8080

:done
pause
