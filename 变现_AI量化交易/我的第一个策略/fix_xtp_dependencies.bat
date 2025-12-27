@echo off
REM 修复 XTP 接口的依赖问题

echo ========================================
echo 修复 XTP 接口依赖
echo ========================================
echo.

REM 激活 conda 环境
echo [步骤 1/2] 正在激活 conda 环境 veighna...
call conda activate veighna
if errorlevel 1 (
    echo 错误: 无法激活 conda 环境 veighna
    pause
    exit /b 1
)
echo ✓ Conda 环境激活成功
echo.

REM 显示当前 Python 路径
python -c "import sys; print(f'Python 路径: {sys.executable}')"
echo.

REM 安装缺失的依赖
echo [步骤 2/2] 正在安装缺失的依赖包...
echo.
echo 安装 importlib_metadata...
pip install importlib_metadata
echo.

echo 验证安装...
python -c "try: import importlib_metadata; print('✓ importlib_metadata 安装成功'); except Exception as e: print(f'✗ 安装失败: {e}')"
echo.

echo ========================================
echo 修复完成！
echo ========================================
echo.
echo 现在请重新运行 run.py 测试 XTP 接口
echo.
pause

