from main import execute_python

result = execute_python("""
print("Hello World!")
x = 1 + 2
print(f"Result: {x}")
""")
print(result)