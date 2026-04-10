from langchain.tools import tool
from langchain_openai import ChatOpenAI, OpenAI
from langchain.chat_models import init_chat_model
from langchain_qwq import ChatQwen
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_agent

from utils.db_tools import get_db_table_info_mysql, get_db_table_structure_mysql



@tool
def call_agent1(query: str, table_name: str):
    """"
    编写数据库增删、改、查的SQL语句。

    参数：
        query: 需要做的事情
    返回：
        result: 数据库增删、改、查代码
    """

    AGENT2_SYSTEM_PROMPT = """
    你是一个专业的数据库增删、改、查的SQL语句编写agent。

    数据库背景信息：
    1. 数据库类型 MySQL，需要使用 MySQL 语法编写 SQL 增删、改、查语句。

    数据库表格结构：
    1. 你需要根据工具获取的表格结构。

    输出格式：
    1. 你只需要返回SQL语句，不要有其他内容，否则将会受到惩罚。
    """

    model = ChatOpenAI(
        model="qwen-plus",
        temperature=0,
        streaming=True
    )

    agent = create_agent(
        model=model,
        tools=[get_db_table_structure_mysql],
        system_prompt=AGENT2_SYSTEM_PROMPT,
    )
    print(f"agent1需要做什么：{query}, 根据表格：{table_name}")

    result = agent.invoke(
        {"messages":
            [
                {"role": "user", "content": query + ', 根据表格：' + table_name},
            ]
        }
    )

    return result['messages'][-1].content



@tool
def call_agent2(query: str):
    """"
    执行数据库增删、改、查的SQL语句
    参数：
        query: 数据库查询语句
    返回：
        result: 数据库查询结果
    """


    AGENT1_SYSTEM_PROMPT = """
    你是一个专业的数据库查询agent。

    你负责使用工具 执行增删、改、查数据库表的功能。

    工具：
    1. get_db_table_info_mysql。
    
    注意：
    不要生成其他信息，也不要提问，只返回数据库返回的信息，否则将会受到惩罚。
    """

    model = ChatOpenAI(
        model="qwen-plus",
        temperature=0,
        streaming=True
    )

    agent = create_agent(
        model=model,
        tools=[get_db_table_info_mysql],
        system_prompt=AGENT1_SYSTEM_PROMPT,
    )

    print(f"agent2需要做什么：{query}")

    result = agent.invoke(
        {"messages":
            [
                {"role": "user", "content": query},
            ]
        }
    )

    print(f"agent2返回的结果：{result['messages'][-1].content}")
    return result['messages'][-1].content
