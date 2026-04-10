import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from utils.subagent import call_agent1, call_agent2

os.environ["OPENAI_API_KEY"] = "sk-5bfb31a9765849beb9c8068fbb24e933"
os.environ['OPENAI_API_BASE'] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

if __name__ == "__main__":

    SUPERVISOR_SYSTEM_PROMPT = """
    你是一个专业的SPA Booking Agent。

    你有2个agent可以使用，分别是：
    1. agent1：编写数据库增删、改、查的SQL语句, 给予agent2使用（绝对不允许、编写删除数据库的SQL语句）。
    1. agent2：根据agent1编写的SQL语句，执行增删、改、查数据库表的功能）。

    拥有的数据库表格名称：
    1. therapists：技师基本信息表

    注意：
    1. agent2的工作需要在，agent1编写的指令代码的基础上执行。
    2. 如果数据库没有对应的数据，不允许自己生成数据，否则将会受到惩罚。
    3. 你不允许做agent1和agent2的工作，否则将会受到惩罚。
    """

    model = ChatOpenAI(
        model="qwen-plus",
        temperature=0,
        streaming=True
    )

    agent = create_agent(
        model=model,
        tools=[call_agent1, call_agent2],
        system_prompt=SUPERVISOR_SYSTEM_PROMPT,
    )

    result = agent.invoke(
        {"messages":
            [
                {"role": "user", "content": "我想获取5号技师的所有信息"},
            ]
         }
    )

    print(result)
