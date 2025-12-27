# XTP 接口问题快速修复卡片

> **快速参考**：遇到 XTP 接口问题时，按此卡片快速排查和修复

---

## 🔍 问题：菜单中没有【连接XTP】

### 快速诊断（30秒）

```bash
conda activate veighna
python -c "from vnpy_xtp import XtpGateway; print('OK')"
```

**如果报错** → 继续下面的修复步骤

---

## 🛠️ 快速修复（3步）

### 步骤 1：检查环境
```bash
conda activate veighna
python -c "import sys; print(sys.executable)"
# 应该显示：D:\Anaconda\envs\veighna\python.exe
```

### 步骤 2：安装/修复
```bash
# 方法 A：使用脚本（推荐）
双击运行：install_xtp_only.bat

# 方法 B：手动安装
call "D:\VisualStudio\VC\Auxiliary\Build\vcvars64.bat"
conda activate veighna
pip install vnpy_xtp
pip install importlib_metadata
```

### 步骤 3：验证
```bash
python run.py
# 查看输出，应该看到：✓ XTP 接口已加载
```

---

## ❌ 常见错误速查

| 错误信息 | 原因 | 解决方法 |
|---------|------|---------|
| `No module named 'vnpy_xtp'` | 未安装或环境不对 | 在 veighna 环境中安装 |
| `No module named 'importlib_metadata'` | 缺少依赖 | `pip install importlib_metadata` |
| `c++: error: /MT` | 使用了 MinGW | 使用 MSVC：`call vcvars64.bat` |
| `Need python for x86` | 使用了 32 位 MSVC | 使用 64 位：`vcvars64.bat` |

---

## ✅ 成功标志

运行 `python run.py` 后，应该看到：

```
[调试] 开始导入 XTP 模块...
✓ XTP 模块导入成功
✓ XTP 接口已加载（A股、ETF期权）
```

菜单中应该出现：**【系统】→ 【连接XTP】**

---

## 📞 需要帮助？

查看详细文档：`VeighNa_XTP接口安装问题排查指南.md`

