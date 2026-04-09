import os
from utils.db_tools import get_therapist_info_mysql, get_db_table_info
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

os.environ["OPENAI_API_KEY"] = "sk-5bfb31a9765849beb9c8068fbb24e933"
os.environ['OPENAI_API_BASE'] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

if __name__ == "__main__":


    SYSTEM_PROMPT = """
    你是一个非常厉害的SPA店长。

    以下是你的个人职责：
    1. 增删、改、查数据库的功能（绝对不允许、删除数据库）。
    2. 顾客预约，可以预定指定时间的技师。
    3. 顾客爽约，可以删除该技师的预定。
    4. 顾客加钟，可以修改该技师的下钟时间 (如果，与后续有客户预约的时间重叠，则不允许修改)。
    5. 只有技师状态为在职的情况下，才可以给技师安排顾客。
    6. 查询技师的信息，通常格式：五号技师、5号技师或技师5号，表示5。

    数据库表格：
    1. therapists：技师基本信息表

    注意
    1. 当遇到不知道的信息时，必须使用工具查询，不可以自己猜测，否则将会受到惩罚。
    2. 当对数据库进行查询时，不要生成其他信息，也不要提问，只返回数据库返回的信息，否则将会受到惩罚。

    """

    model = ChatOpenAI(
        model="qwen-plus",
        temperature=0,
        streaming=True
    )

    agent = create_agent(
        model=model,
        tools=[get_therapist_info_mysql, get_db_table_info],
        system_prompt=SYSTEM_PROMPT,
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "我想获取技师基本信息表的结构"}]}
    )
    print(result)
    