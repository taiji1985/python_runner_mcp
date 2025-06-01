# Python Runner MCP Server

[English](README.md) | [中文](README.zh-CN.md)

一个基于 FastMCP 框架的 Python 代码执行服务器，专为数据科学和机器学习工作流程设计。

## 🚀 特性

- **安全的 Python 代码执行**：在隔离的命名空间中执行 Python 代码
- **丰富的数据科学库**：预装了常用的数据科学和机器学习包
- **实时输出捕获**：捕获标准输出、错误输出和返回值
- **MCP 协议支持**：完全兼容 Model Context Protocol
- **简单易用**：提供简洁的 API 接口

## ⚡ 快速开始（推荐）

使用 `uvx` 可以直接运行服务器，无需安装：

```bash
# 直接运行（无需预先安装）
uvx python-runner
```

### 在 Claude Desktop 中配置

在 Claude Desktop 的配置文件中添加以下配置：

```json
{
  "mcpServers": {
    "python-runner": {
      "command": "uvx",
      "args": ["python-runner"]
    }
  }
}
```

**配置文件位置：**
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

### 调试和测试

```bash
# 使用 MCP Inspector 调试服务器
npx @modelcontextprotocol/inspector uvx python-runner
```

## 📦 预装库

本项目预装了以下常用的数据科学和机器学习库：

- **数据处理**: `numpy`, `pandas`, `scipy`
- **机器学习**: `scikit-learn`
- **数据可视化**: `matplotlib`, `seaborn`, `plotly`
- **科学计算**: `sympy`
- **图像处理**: `pillow`
- **网络分析**: `networkx`
- **交互式开发**: `jupyter`
- **HTTP 请求**: `requests`

## 🛠️ 手动安装（开发用途）

### 前置要求

- Python 3.12 或更高版本
- uv 包管理器（推荐）或 pip

### 使用 uv 安装（推荐）

```bash
# 克隆仓库
git clone <repository-url>
cd python_runner

# 安装依赖
uv sync
```

### 使用 pip 安装

```bash
# 克隆仓库
git clone <repository-url>
cd python_runner

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -e .
```

## 🚀 使用方法

### 在 Claude Desktop 中使用

配置完成后，你可以在 Claude Desktop 中直接使用 Python 代码执行功能：

```
请帮我执行这段 Python 代码：

import pandas as pd
import numpy as np

# 创建示例数据
data = pd.DataFrame({
    'sales': [100, 150, 200, 120, 180],
    'profit': [20, 30, 50, 25, 40]
})

print("销售数据统计：")
print(data.describe())
```

### 作为独立 MCP 服务器运行

```bash
# 使用 uvx（推荐）
uvx python-runner

# 或者本地运行
python main.py
```

### 直接调用 API

```python
from main import execute_python

# 执行简单的 Python 代码
result = execute_python("""
print("Hello, World!")
x = 1 + 2
print(f"Result: {x}")
""")

print(result)
# 输出:
# {
#     'output': 'Hello, World!\nResult: 3\n',
#     'error': '',
#     'success': True
# }
```

### 数据科学示例

```python
# 数据分析示例
code = """
import pandas as pd
import numpy as np

# 创建示例数据
data = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100)
})

print(f"数据形状: {data.shape}")
print(f"统计摘要:\n{data.describe()}")
"""

result = execute_python(code)
print(result['output'])
```

```python
# 机器学习示例
code = """
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 生成示例数据
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# 预测和评估
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.4f}")
"""

result = execute_python(code)
print(result['output'])
```

## 📋 API 参考

### `execute_python(code: str) -> dict`

执行 Python 代码并返回结果。

**参数:**
- `code` (str): 要执行的 Python 代码字符串

**返回值:**
- `dict`: 包含以下键的字典：
  - `output` (str): 标准输出内容
  - `error` (str): 错误信息（如果有）
  - `success` (bool): 执行是否成功

## 🔒 安全性

- 代码在隔离的命名空间中执行
- 捕获并重定向标准输出和错误输出
- 提供详细的错误追踪信息

⚠️ **注意**: 此工具执行任意 Python 代码，请确保在受信任的环境中使用。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [FastMCP](https://github.com/jlowin/fastmcp) - 优秀的 MCP 框架
- 所有预装的开源数据科学库的维护者们

---

**Made with ❤️ for the data science community**