@echo off
REM VeighNa 所有接口安装脚本
REM 自动设置 MSVC 编译环境并安装所有 VeighNa 交易接口

echo ========================================
echo VeighNa 所有接口安装工具
echo ========================================
echo.

REM 设置 MSVC 编译环境
echo [步骤 1/2] 正在设置 MSVC 编译环境...
call "D:\VisualStudio\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 (
    echo 错误: 无法找到 MSVC 编译环境
    pause
    exit /b 1
)
echo ✓ MSVC 环境设置成功
echo.

REM 激活 conda 环境
echo [步骤 2/2] 正在激活 conda 环境 veighna...
call conda activate veighna
if errorlevel 1 (
    echo 错误: 无法激活 conda 环境 veighna
    pause
    exit /b 1
)
echo ✓ Conda 环境激活成功
echo.

echo ========================================
echo 开始安装接口...
echo ========================================
echo.

REM 国内期货接口
echo [1] 安装 CTP 接口（国内期货）...
pip install vnpy_ctp
echo.

echo [2] 安装 CTP Mini 接口...
pip install vnpy_mini
echo.

echo [3] 安装飞马接口...
pip install vnpy_femas
echo.

echo [4] 安装恒生UFT接口...
pip install vnpy_uft
echo.

REM 国内股票接口
echo [5] 安装 XTP 接口（A股）...
pip install vnpy_xtp
echo.

echo [6] 安装华鑫奇点接口（A股）...
pip install vnpy_tora
echo.

echo [7] 安装东证OST接口...
pip install vnpy_ost
echo.

echo [8] 安装东方财富EMT接口...
pip install vnpy_emt
echo.

REM 期权接口
echo [9] 安装 CTP期权接口...
pip install vnpy_sopt
echo.

echo [10] 安装顶点HTS接口...
pip install vnpy_hts
echo.

REM ===== 海外市场接口（重点）=====
echo ========================================
echo 安装海外市场接口（港股、美股）
echo ========================================
echo.

echo [11] 安装 Interactive Brokers 接口（美股、港股）⭐...
pip install vnpy_ib
echo.

echo [12] 安装易盛9.0外盘接口...
pip install vnpy_tap
echo.

echo [13] 安装直达期货接口...
pip install vnpy_da
echo.

REM 其他接口
echo [14] 安装 TTS 仿真接口...
pip install vnpy_tts
echo.

echo ========================================
echo 安装完成！
echo ========================================
echo.
echo 提示：
echo - Interactive Brokers (IB) 是交易美股和港股的主要接口
echo - 需要先在 IB 官网开户：https://www.interactivebrokers.com
echo - 其他接口根据你的需求选择性使用
echo.
pause

