@echo off
REM 在 veighna 环境中安装 XTP 接口

echo ========================================
echo 在 veighna 环境中安装 XTP 接口
echo ========================================
echo.

REM 设置 MSVC 编译环境
echo [步骤 1/3] 正在设置 MSVC 编译环境...
call "D:\VisualStudio\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 (
    echo 错误: 无法找到 MSVC 编译环境
    pause
    exit /b 1
)
echo ✓ MSVC 环境设置成功
echo.

REM 激活 conda 环境
echo [步骤 2/3] 正在激活 conda 环境 veighna...
call conda activate veighna
if errorlevel 1 (
    echo 错误: 无法激活 conda 环境 veighna
    pause
    exit /b 1
)
echo ✓ Conda 环境激活成功
echo.

REM 显示当前 Python 路径
echo [步骤 2.5] 检查当前 Python 环境...
python -c "import sys; print(f'Python 路径: {sys.executable}')"
echo.

REM 安装 XTP 接口
echo [步骤 3/3] 正在安装 XTP 接口...
pip install vnpy_xtp
if errorlevel 1 (
    echo.
    echo 错误: XTP 安装失败
    pause
    exit /b 1
)
echo.

echo ========================================
echo 安装完成！
echo ========================================
echo.
echo 验证安装...
python -c "try: from vnpy_xtp import XtpGateway; print('✓ XTP 接口安装成功，可以正常导入'); except Exception as e: print(f'✗ XTP 接口导入失败: {e}')"
echo.
pause

