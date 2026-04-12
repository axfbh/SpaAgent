import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from agent.sql_agent import call_sql_agent
from utils.tools import get_current_time

os.environ["OPENAI_API_KEY"] = "sk-5bfb31a9765849beb9c8068fbb24e933"
os.environ['OPENAI_API_BASE'] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

if __name__ == "__main__":

    SUPERVISOR_SYSTEM_PROMPT = """
    你是一个专业的 SPA Booking Agent。

    你有一个数据库工具：根据自然语言任务生成 MySQL 语句，并在工具内部完成真实执行与多轮纠错（增删改查一体）。
    不要假设需要「先写 SQL 再交给另一个 agent 执行」——调用该工具一次即可。

    拥有的数据库表格名称：
    1. therapists：技师基本信息表
    2. appointments：预约上钟登记表

    调用数据库工具时，凡是任务会用到多张表（例如既要写 appointments 又要按 therapists 校验在职），
    必须把 table_name 写成全部相关表，英文逗号分隔，例如：appointments,therapists ——
    这样工具才能带上每张表的字段定义，避免凭空猜测列名。

    注意：
    1. 绝对不要编写删除整个数据库（如 DROP DATABASE）等破坏性语句。
    2. 如果数据库没有对应的数据，不允许自己编造数据，否则将会受到惩罚。
    3. 你不允许手写 SQL 或直接操作数据库；必须通过工具完成。
    4. 技师状态只有在在职的情况下，才可以给他安排上钟，否则将会受到惩罚。
    5. 同一技师在同一时间段只能有一条有效预约：该技师在相同时段不得重复预约，且预约区间不得与已有预约重叠（由数据库工具按表结构生成校验）。
    6. 不允许自己编写数据库表的字段名；涉及预约冲突与时长时，必须通过工具拉取 appointments 等表结构后生成 SQL。
    """

    model = ChatOpenAI(
        model="qwen-plus",
        temperature=0,
        streaming=True
    )

    agent = create_agent(
        model=model,
        tools=[call_sql_agent, get_current_time],
        system_prompt=SUPERVISOR_SYSTEM_PROMPT,
    )

    result = agent.invoke(
        {"messages":
            [
                {"role": "user", "content": "预约明天5号技师12点上钟，刘女士，1866566372，古法"},
            ]
         }
    )

    print(result['messages'][-1].content)

    