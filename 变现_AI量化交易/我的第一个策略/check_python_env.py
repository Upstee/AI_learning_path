"""检查当前 Python 环境"""
import sys
import os

print("=" * 50)
print("检查当前 Python 环境")
print("=" * 50)
print(f"Python 版本: {sys.version}")
print(f"Python 路径: {sys.executable}")
print(f"Python 可执行文件: {sys.executable}")
print()

# 检查 conda 环境
conda_env = os.environ.get('CONDA_DEFAULT_ENV', '未设置')
print(f"Conda 环境: {conda_env}")
print(f"CONDA_PREFIX: {os.environ.get('CONDA_PREFIX', '未设置')}")
print()

# 检查 vnpy_xtp
print("检查 vnpy_xtp...")
try:
    import vnpy_xtp
    print(f"✓ vnpy_xtp 已安装")
    print(f"  路径: {vnpy_xtp.__file__}")
except ImportError as e:
    print(f"✗ vnpy_xtp 未安装: {e}")

print()
print("检查 vnpy...")
try:
    import vnpy
    print(f"✓ vnpy 已安装")
    print(f"  路径: {vnpy.__file__}")
except ImportError as e:
    print(f"✗ vnpy 未安装: {e}")

print()
print("=" * 50)

