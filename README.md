# Python Runner MCP Server

[English](README.md) | [中文](README.zh-CN.md)

A Python code execution server based on the FastMCP framework, designed specifically for data science and machine learning workflows.

## 🚀 Features

- **Safe Python Code Execution**: Execute Python code in isolated namespaces
- **Rich Data Science Libraries**: Pre-installed with commonly used data science and machine learning packages
- **Real-time Output Capture**: Capture standard output, error output, and return values
- **MCP Protocol Support**: Fully compatible with Model Context Protocol
- **Easy to Use**: Provides a simple and clean API interface

## ⚡ Quick Start (Recommended)

Use `uvx` to run the server directly without installation:

```bash
# Run directly (no pre-installation required)
uvx python-runner
```

### Configure in Claude Desktop

Add the following configuration to your Claude Desktop config file:

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

**Config file locations:**
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

### Debug and Test

```bash
# Use MCP Inspector to debug the server
npx @modelcontextprotocol/inspector uvx python-runner
```

## 📦 Pre-installed Libraries

This project comes pre-installed with the following commonly used data science and machine learning libraries:

- **Data Processing**: `numpy`, `pandas`, `scipy`
- **Machine Learning**: `scikit-learn`
- **Data Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Scientific Computing**: `sympy`
- **Image Processing**: `pillow`
- **Network Analysis**: `networkx`
- **Interactive Development**: `jupyter`
- **HTTP Requests**: `requests`

## 🛠️ Manual Installation (For Development)

### Prerequisites

- Python 3.12 or higher
- uv package manager (recommended) or pip

### Install with uv (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd python_runner

# Install dependencies
uv sync
```

### Install with pip

```bash
# Clone the repository
git clone <repository-url>
cd python_runner

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## 🚀 Usage

### Using with Claude Desktop

After configuration, you can directly use Python code execution functionality in Claude Desktop:

```
Please help me execute this Python code:

import pandas as pd
import numpy as np

# Create sample data
data = pd.DataFrame({
    'sales': [100, 150, 200, 120, 180],
    'profit': [20, 30, 50, 25, 40]
})

print("Sales data statistics:")
print(data.describe())
```

### Running as Standalone MCP Server

```bash
# Using uvx (recommended)
uvx python-runner

# Or run locally
python main.py
```

### Direct API Call

```python
from main import execute_python

# Execute simple Python code
result = execute_python("""
print("Hello, World!")
x = 1 + 2
print(f"Result: {x}")
""")

print(result)
# Output:
# {
#     'output': 'Hello, World!\nResult: 3\n',
#     'error': '',
#     'success': True
# }
```

### Data Science Examples

```python
# Data analysis example
code = """
import pandas as pd
import numpy as np

# Create sample data
data = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100)
})

print(f"Data shape: {data.shape}")
print(f"Statistical summary:\n{data.describe()}")
"""

result = execute_python(code)
print(result['output'])
```

```python
# Machine learning example
code = """
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")
"""

result = execute_python(code)
print(result['output'])
```

## 📋 API Reference

### `execute_python(code: str) -> dict`

Execute Python code and return the result.

**Parameters:**
- `code` (str): Python code string to execute

**Returns:**
- `dict`: Dictionary containing the following keys:
  - `output` (str): Standard output content
  - `error` (str): Error message (if any)
  - `success` (bool): Whether execution was successful

## 🔒 Security

- Code executes in isolated namespaces
- Captures and redirects standard output and error output
- Provides detailed error tracking information

⚠️ **Warning**: This tool executes arbitrary Python code. Please ensure you use it in a trusted environment.

## 🤝 Contributing

Issues and Pull Requests are welcome!

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FastMCP](https://github.com/jlowin/fastmcp) - Excellent MCP framework
- All maintainers of the pre-installed open-source data science libraries

---

**Made with ❤️ for the data science community**