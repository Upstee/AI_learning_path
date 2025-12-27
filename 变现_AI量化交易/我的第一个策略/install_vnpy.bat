@echo off
REM VeighNa 接口安装脚本
REM 自动设置 MSVC 编译环境并安装 VeighNa 接口

echo ========================================
echo VeighNa 接口安装工具
echo ========================================
echo.

REM 设置 MSVC 编译环境
echo [1/3] 正在设置 MSVC 编译环境...
call "D:\VisualStudio\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 (
    echo 错误: 无法找到 MSVC 编译环境
    echo 请检查路径: D:\VisualStudio\VC\Auxiliary\Build\vcvars64.bat
    pause
    exit /b 1
)
echo ✓ MSVC 环境设置成功
echo.

REM 激活 conda 环境
echo [2/3] 正在激活 conda 环境 veighna...
call conda activate veighna
if errorlevel 1 (
    echo 错误: 无法激活 conda 环境 veighna
    echo 请确保已创建该环境
    pause
    exit /b 1
)
echo ✓ Conda 环境激活成功
echo.

REM 安装指定的包
echo [3/3] 正在安装: %*
if "%*"=="" (
    echo 错误: 请指定要安装的包名
    echo 用法: install_vnpy.bat vnpy_xtp
    echo 示例: install_vnpy.bat vnpy_xtp
    pause
    exit /b 1
)
echo.
pip install %*
if errorlevel 1 (
    echo.
    echo 安装失败！
    pause
    exit /b 1
)

echo.
echo ========================================
echo 安装完成！
echo ========================================
pause

