from langchain.tools import tool
from langchain_openai import ChatOpenAI, OpenAI
from langchain.chat_models import init_chat_model
from langchain_qwq import ChatQwen

from utils.db_tools import get_db_table_info_mysql, get_db_table_structure_mysql

@tool
def call_agent1(query: str):
    """"
    调用agent1，执行数据库增删、改、查操作
    参数：
        query: 数据库查询语句
    返回：
        result: 数据库查询结果
    """


    AGENT1_SYSTEM_PROMPT = """
    你是一个专业的数据库查询agent。

    你负责执行增删、改、查数据库表的功能。
    
    注意：
    不要生成其他信息，也不要提问，只返回数据库返回的信息，否则将会受到惩罚。
    """

    model = ChatQwen(
        model="qwen-plus",
        temperature=0,
        streaming=False
    )

    model.bind_tools([get_db_table_info_mysql])

    result = model.invoke(
        {"messages": 
            [
                {"role": "system", "content": AGENT1_SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ]
        }
    )

    return result["messages"][-1].content



@tool
def call_agent2(query: str):
    """"
    调用agent2，编写数据库增删、改、查的代码

    参数：
        query: 需要编写的数据库代码，需要的相关信息
    返回：
        result: 数据库增删、改、查代码
    """

    AGENT2_SYSTEM_PROMPT = """
    你是一个专业的数据库增删、改、查代码编写agent。

    你负责编写数据库增删、改、查代码（绝对不允许、编写删除数据库）。首先，
    
    具体步骤：
    1. 你需要根据表格名称，获取表格结构信息
    2. 根据表格结构信息，编写数据库增删、改、查代码
    
    注意：
    不要生成其他信息，也不要提问，只返回数据库返回的信息，否则将会受到惩罚。
    """

    model = ChatQwen(
        model="qwen-plus",
        temperature=0,
        streaming=False
    )

    model.bind_tools([get_db_table_structure_mysql])

    result = model.invoke(
        {"messages": 
            [
                {"role": "system", "content": AGENT2_SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ]
        }
    )

    return result["messages"][-1].content
