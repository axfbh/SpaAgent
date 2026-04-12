from langchain.tools import tool
import time

@tool
def get_current_time():
    """获取当前时间"""
    print("get_current_time工具被调用")
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
