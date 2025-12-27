"""在 veighna 环境中测试 XTP 导入"""
import sys
import os

print("=" * 50)
print("在 veighna 环境中测试 XTP 导入")
print("=" * 50)
print(f"Python 版本: {sys.version}")
print(f"Python 路径: {sys.executable}")
print(f"Conda 环境: {os.environ.get('CONDA_DEFAULT_ENV', '未设置')}")
print()

print("[1] 测试导入 vnpy_xtp 模块...")
try:
    import vnpy_xtp
    print(f"✓ vnpy_xtp 模块导入成功")
    print(f"  模块路径: {vnpy_xtp.__file__}")
except Exception as e:
    print(f"✗ vnpy_xtp 模块导入失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("[2] 测试导入 XtpGateway...")
try:
    from vnpy_xtp import XtpGateway
    print(f"✓ XtpGateway 导入成功")
    print(f"  类名: {XtpGateway.__name__}")
    print(f"  默认名称: {getattr(XtpGateway, 'default_name', 'N/A')}")
except Exception as e:
    print(f"✗ XtpGateway 导入失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("[3] 测试创建 XtpGateway 实例（不连接）...")
try:
    from vnpy.event import EventEngine
    event_engine = EventEngine()
    gateway = XtpGateway(event_engine, "XTP")
    print(f"✓ XtpGateway 实例创建成功")
    print(f"  网关名称: {gateway.gateway_name}")
except Exception as e:
    print(f"✗ XtpGateway 实例创建失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 50)
print("✓ 所有测试通过！XTP 接口可以正常使用")
print("=" * 50)

