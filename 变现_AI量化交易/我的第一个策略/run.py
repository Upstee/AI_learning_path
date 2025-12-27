import os
from pathlib import Path

# ===== 关键：在导入 vnpy 之前设置工作目录 =====
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

vntrader_dir = script_dir / ".vntrader"
vntrader_dir.mkdir(exist_ok=True)

strategies_dir = script_dir / "strategies"
strategies_dir.mkdir(exist_ok=True)

print(f"工作目录: {script_dir}")

from vnpy.event import EventEngine
from vnpy.trader.engine import MainEngine
from vnpy.trader.ui import MainWindow, create_qapp

from vnpy_ctastrategy import CtaStrategyApp
from vnpy_ctabacktester import CtaBacktesterApp
from vnpy_datamanager import DataManagerApp

# ===== 添加所有交易接口 =====
# 国内期货接口
try:
    from vnpy_ctp import CtpGateway
    CTP_AVAILABLE = True
except ImportError:
    CTP_AVAILABLE = False

try:
    from vnpy_mini import MiniGateway
    MINI_AVAILABLE = True
except ImportError:
    MINI_AVAILABLE = False

try:
    from vnpy_femas import FemasGateway
    FEMAS_AVAILABLE = True
except ImportError:
    FEMAS_AVAILABLE = False

try:
    from vnpy_uft import UftGateway
    UFT_AVAILABLE = True
except ImportError:
    UFT_AVAILABLE = False

# 国内股票接口（A股）
print("\n[调试] 开始导入 XTP 模块...")
try:
    import vnpy_xtp
    print(f"[调试] vnpy_xtp 模块路径: {vnpy_xtp.__file__}")
    from vnpy_xtp import XtpGateway
    XTP_AVAILABLE = True
    print("✓ XTP 模块导入成功")
except ImportError as e:
    XTP_AVAILABLE = False
    print(f"✗ XTP 模块导入失败（ImportError）: {e}")
    import sys
    print(f"[调试] Python 路径: {sys.path}")
except Exception as e:
    XTP_AVAILABLE = False
    print(f"✗ XTP 模块导入出错（其他异常）: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

try:
    from vnpy_tora import ToraStockGateway, ToraOptionGateway
    TORA_AVAILABLE = True
except ImportError:
    TORA_AVAILABLE = False

try:
    from vnpy_ost import OstGateway
    OST_AVAILABLE = True
except ImportError:
    OST_AVAILABLE = False

try:
    from vnpy_emt import EmtGateway
    EMT_AVAILABLE = True
except ImportError:
    EMT_AVAILABLE = False

# 期权接口
try:
    from vnpy_sopt import SoptGateway
    SOPT_AVAILABLE = True
except ImportError:
    SOPT_AVAILABLE = False

try:
    from vnpy_hts import HtsGateway
    HTS_AVAILABLE = True
except ImportError:
    HTS_AVAILABLE = False

# ===== 海外市场接口（港股、美股）=====
# Interactive Brokers - 支持美股、港股、全球市场
try:
    from vnpy_ib import IbGateway
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False

# 外盘期货接口
try:
    from vnpy_tap import TapGateway
    TAP_AVAILABLE = True
except ImportError:
    TAP_AVAILABLE = False

try:
    from vnpy_da import DaGateway
    DA_AVAILABLE = True
except ImportError:
    DA_AVAILABLE = False

# 其他接口
try:
    from vnpy_tts import TtsGateway
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


def main():
    """启动 VeighNa Trader"""
    qapp = create_qapp()
    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)
    
    # ===== 添加交易接口 =====
    print("=" * 50)
    print("正在加载交易接口...")
    print("=" * 50)
    
    # 国内期货接口
    if CTP_AVAILABLE:
        main_engine.add_gateway(CtpGateway)
        print("✓ CTP 接口已加载（国内期货）")
    
    if MINI_AVAILABLE:
        main_engine.add_gateway(MiniGateway)
        print("✓ CTP Mini 接口已加载（国内期货）")
    
    if FEMAS_AVAILABLE:
        main_engine.add_gateway(FemasGateway)
        print("✓ 飞马接口已加载（国内期货）")
    
    if UFT_AVAILABLE:
        main_engine.add_gateway(UftGateway)
        print("✓ 恒生UFT接口已加载（国内期货、ETF期权）")
    
    # 国内股票接口（A股）
    if XTP_AVAILABLE:
        try:
            main_engine.add_gateway(XtpGateway)
            print("✓ XTP 接口已加载（A股、ETF期权）")
        except Exception as e:
            print(f"✗ XTP 接口加载失败: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("✗ XTP 接口不可用（模块未安装或导入失败）")
    
    if TORA_AVAILABLE:
        main_engine.add_gateway(ToraStockGateway)
        main_engine.add_gateway(ToraOptionGateway)
        print("✓ 华鑫奇点接口已加载（A股、ETF期权）")
    
    if OST_AVAILABLE:
        main_engine.add_gateway(OstGateway)
        print("✓ 东证OST接口已加载（A股）")
    
    if EMT_AVAILABLE:
        main_engine.add_gateway(EmtGateway)
        print("✓ 东方财富EMT接口已加载（A股）")
    
    # 期权接口
    if SOPT_AVAILABLE:
        main_engine.add_gateway(SoptGateway)
        print("✓ CTP期权接口已加载（ETF期权）")
    
    if HTS_AVAILABLE:
        main_engine.add_gateway(HtsGateway)
        print("✓ 顶点HTS接口已加载（ETF期权）")
    
    # ===== 海外市场接口（重点：港股、美股）=====
    if IB_AVAILABLE:
        main_engine.add_gateway(IbGateway)
        print("✓ Interactive Brokers 接口已加载（美股、港股、全球市场）⭐")
    
    if TAP_AVAILABLE:
        main_engine.add_gateway(TapGateway)
        print("✓ 易盛9.0外盘接口已加载（外盘期货）")
    
    if DA_AVAILABLE:
        main_engine.add_gateway(DaGateway)
        print("✓ 直达期货接口已加载（外盘期货）")
    
    # 其他接口
    if TTS_AVAILABLE:
        main_engine.add_gateway(TtsGateway)
        print("✓ TTS接口已加载（期货仿真）")
    
    print("=" * 50)
    
    # 添加功能模块
    main_engine.add_app(CtaStrategyApp)
    main_engine.add_app(CtaBacktesterApp)
    main_engine.add_app(DataManagerApp)
    
    main_window = MainWindow(main_engine, event_engine)
    main_window.showMaximized()
    
    qapp.exec()


if __name__ == "__main__":
    main()