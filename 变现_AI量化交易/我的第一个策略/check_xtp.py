"""检查 XTP 接口安装状态"""
import sys

print("=" * 50)
print("检查 XTP 接口安装状态")
print("=" * 50)
print(f"Python 版本: {sys.version}")
print(f"Python 路径: {sys.executable}")
print()

# 检查是否安装了 vnpy_xtp
print("[1] 检查 vnpy_xtp 包是否安装...")
try:
    import pkg_resources
    installed_packages = [d.project_name for d in pkg_resources.working_set]
    if 'vnpy-xtp' in installed_packages or 'vnpy_xtp' in installed_packages:
        print("✓ vnpy_xtp 包已安装")
        # 获取版本信息
        try:
            dist = pkg_resources.get_distribution('vnpy-xtp')
            print(f"  版本: {dist.version}")
            print(f"  位置: {dist.location}")
        except:
            try:
                dist = pkg_resources.get_distribution('vnpy_xtp')
                print(f"  版本: {dist.version}")
                print(f"  位置: {dist.location}")
            except:
                print("  无法获取版本信息")
    else:
        print("✗ vnpy_xtp 包未安装")
        print("  已安装的相关包:")
        for pkg in installed_packages:
            if 'xtp' in pkg.lower() or 'vnpy' in pkg.lower():
                print(f"    - {pkg}")
except Exception as e:
    print(f"✗ 检查失败: {e}")

print()
print("[2] 尝试导入 vnpy_xtp 模块...")
try:
    import vnpy_xtp
    print(f"✓ vnpy_xtp 模块导入成功")
    print(f"  模块路径: {vnpy_xtp.__file__}")
    print(f"  模块属性: {dir(vnpy_xtp)}")
except ImportError as e:
    print(f"✗ 导入失败（ImportError）: {e}")
    print(f"  错误详情: {str(e)}")
except Exception as e:
    print(f"✗ 导入失败（其他异常）: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print()
print("[3] 尝试导入 XtpGateway...")
try:
    from vnpy_xtp import XtpGateway
    print(f"✓ XtpGateway 导入成功")
    print(f"  类名: {XtpGateway.__name__}")
    print(f"  默认名称: {getattr(XtpGateway, 'default_name', 'N/A')}")
except ImportError as e:
    print(f"✗ 导入失败（ImportError）: {e}")
except Exception as e:
    print(f"✗ 导入失败（其他异常）: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 50)
print("检查完成")
print("=" * 50)

