# BRK-B 数据展示中心 - PowerShell 启动脚本
# 用法: .\start_server.ps1

$Host.UI.RawUI.BackgroundColor = "Black"
$Host.UI.RawUI.ForegroundColor = "White"
Clear-Host

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  BRK-B 数据展示中心 - 本地服务器启动器" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

function Find-Python {
    # 1. 尝试 python
    $py = Get-Command python -ErrorAction SilentlyContinue
    if ($py) { return $py.Source }

    # 2. 尝试 py
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) { return $py.Source }

    # 3. 尝试 python3
    $py = Get-Command python3 -ErrorAction SilentlyContinue
    if ($py) { return $py.Source }

    # 4. 常见安装路径
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python39\python.exe",
        "C:\Program Files\Python313\python.exe",
        "C:\Program Files\Python312\python.exe",
        "C:\Program Files\Python311\python.exe",
        "C:\Program Files\Python310\python.exe",
        "C:\Program Files\Python39\python.exe",
        "C:\Python313\python.exe",
        "C:\Python312\python.exe",
        "C:\Python311\python.exe",
        "C:\Python310\python.exe",
        "C:\Python39\python.exe",
        "$env:LOCALAPPDATA\Microsoft\WindowsApps\python.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }

    # 5. 尝试 uv
    $uv = "$env:USERPROFILE\.local\bin\uv.exe"
    if (Test-Path $uv) {
        Write-Host "检测到 uv，尝试通过 uv 运行 Python..." -ForegroundColor Yellow
        return "uv:run"
    }

    return $null
}

function Find-FreePort($start = 8080) {
    $port = $start
    while ($true) {
        $listener = New-Object System.Net.Sockets.TcpListener ([System.Net.IPAddress]::Loopback, $port)
        try {
            $listener.Start()
            $listener.Stop()
            return $port
        } catch {
            $port++
        }
    }
}

$pythonPath = Find-Python

if (-not $pythonPath) {
    Write-Host ""
    Write-Host "[错误] 未找到 Python！" -ForegroundColor Red
    Write-Host ""
    Write-Host "请尝试以下方法之一：" -ForegroundColor Yellow
    Write-Host "1. 安装 Python 3（https://python.org）并勾选 'Add to PATH'" -ForegroundColor White
    Write-Host "2. 如果你知道 python.exe 的位置，可以直接运行：" -ForegroundColor White
    Write-Host "   <完整路径>\python.exe -m http.server 8080" -ForegroundColor Gray
    Write-Host "3. 安装 uv（https://docs.astral.sh/uv）后重试" -ForegroundColor White
    Write-Host ""
    pause
    exit 1
}

# 切换到项目根目录
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$rootDir = Split-Path -Parent $scriptDir
Set-Location $rootDir

$port = Find-FreePort 8080

Write-Host "找到运行环境，正在启动服务器..." -ForegroundColor Green
Write-Host ""
Write-Host "  访问地址: http://localhost:$port/brk_b_dashboard/" -ForegroundColor Green
Write-Host "  项目目录: $rootDir" -ForegroundColor Gray
Write-Host ""
Write-Host "  按 Ctrl+C 停止服务器" -ForegroundColor Gray
Write-Host ""

# 自动打开浏览器
Start-Sleep -Seconds 1
Start-Process "http://localhost:$port/brk_b_dashboard/"

# 启动服务器
if ($pythonPath -eq "uv:run") {
    & "$env:USERPROFILE\.local\bin\uv.exe" run python -m http.server $port
} else {
    & $pythonPath -m http.server $port
}
