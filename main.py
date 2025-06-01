from fastmcp import FastMCP
import sys
from io import StringIO
import traceback

mcp = FastMCP("Python Runner")


@mcp.tool()
def execute_python(code: str) -> dict:
    """
    执行Python代码并返回执行结果
    
    Args:
        code: Python代码字符串
        
    Returns:
        包含执行结果的字典，包括：
        - output: 标准输出内容
        - error: 如果有错误，返回错误信息
        - success: 是否执行成功
    """
    # 保存原始的标准输出和错误输出
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    # 创建字符串缓冲区来捕获输出
    stdout_capture = StringIO()
    stderr_capture = StringIO()
    
    result = {
        'output': '',
        'error': '',
        'success': True
    }
    
    try:
        # 重定向标准输出和错误输出
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        
        # 在独立的命名空间中执行代码
        exec_globals = {}
        # 尝试获取表达式的返回值
        return_value =None
        try:
            return_value = eval(code, exec_globals)
            #if return_value is not None:
            #    print(return_value)  # 打印返回值
        except Exception as e:
            # 如果不是表达式，则使用exec执行
            exec(code, exec_globals)
        
        # 获取输出
        result['output'] = str(stdout_capture.getvalue())
        if return_value is not None:
            result['output'] += str(return_value)
        
    except Exception as e:
        result['success'] = False
        result['error'] = f"{str(e)}\n{traceback.format_exc()}"
    
    finally:
        # 恢复原始的标准输出和错误输出
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        # 如果有错误输出，添加到结果中
        stderr_output = stderr_capture.getvalue()
        if stderr_output:
            result['error'] = (result['error'] + '\n' + stderr_output) if result['error'] else stderr_output
    
    return result


def main():
    mcp.run()

if __name__ == "__main__":
    # main()
    r=execute_python("print(5)\n2+2")
    print(r)