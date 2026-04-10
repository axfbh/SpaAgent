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
    1. agent1：根据agent2编写的代码，执行增删、改、查数据库表的功能）。
    2. agent2：编写数据库增删、改、查的代码, 给予agent1使用（绝对不允许、编写删除数据库）。

    数据库表格名称：
    1. therapists：技师基本信息表
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
                {"role": "user", "content": "我想获取技师基本信息表的结构"},

            ]
        }
    )
    print(result)
    