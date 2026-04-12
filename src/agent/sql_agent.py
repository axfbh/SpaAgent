import re
from functools import lru_cache
from typing import Annotated, Literal, TypedDict

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from utils.db_tools import (
    get_db_table_info_mysql,
    get_db_table_structure_mysql,
    run_write_sql,
)


class SQLReflectionState(TypedDict):
    """反思循环：任务、表结构、当前 SQL、执行结果、迭代次数。"""

    task: str
    table_schema: str
    draft: str
    execution_ok: bool
    execution_message: str
    iteration: int
    messages: Annotated[list, add_messages]


class SQLDraft(BaseModel):
    """结构化输出单条可执行 SQL。"""

    sql: str = Field(description="完整、可执行的 MySQL 语句，不要 Markdown 或解释文字")


SQL_REFLECTION_WRITE_SYSTEM = """你是 MySQL 专家，根据任务与表结构编写一条可执行的 SQL（查询与增删改）。

规则：
1. 仅使用表结构中存在的库表与字段名；「技师编号」在 therapists 表中必须对应表结构里的主键/编号列（常见为 `id`），不要用 appointments 的外键名去猜 therapists 的列名。
2. 符合 MySQL 语法；需要时用反引号包裹标识符。
3. 下一条「验证」步骤会真实执行该语句：SELECT/SHOW 等走查询；INSERT/UPDATE/DELETE 会写入数据库并提交。
4. 任务要求「仅在职可预约」时：用一条 `INSERT INTO ... SELECT ... FROM therapists WHERE ... AND status='在职'`；验证时若插入 0 行会判为失败（技师非在职则不会写入）。
5. 同一技师在同一自然日内预约时段不得重复、时段不得重叠：INSERT 的 SELECT 必须带 `AND NOT EXISTS (...)`（或等价逻辑），排除该技师在该日已存在且与新预约时间区间重叠的记录。区间用 `TIMESTAMP(appointment_date, appointment_time)` 为起点；终点 = 起点 + 时长分钟数——时长须来自 appointments / 服务相关表中真实存在的字段（如 `duration_minutes`、服务时长等），**禁止臆造列名**；若表结构仅有开始时刻、无任何时长/结束时刻字段，则至少禁止相同 `therapist_id` + `appointment_date` + `appointment_time` 的第二条约（完全相同时段视为冲突）。已取消类状态若表中有对应字段，可在 NOT EXISTS 子查询中排除。
6. 重叠判定（有起止或有时长时）：新区间 [新开始, 新结束) 与已有区间 [旧开始, 旧结束) 满足 `新开始 < 旧结束 AND 旧开始 < 新结束` 即视为重叠，须在 NOT EXISTS 中写完整条件。
7. 若上一轮 SQL 执行失败，必须根据错误信息修正后给出全新 SQL，不要重复无效语句。
8. 禁止 DROP DATABASE、TRUNCATE 整库等破坏性操作（除非任务明确要求且你确认表名无误）。
"""


def _normalize_sql(text: str) -> str:
    t = (text or "").strip()
    fence = re.search(r"```(?:sql)?\s*([\s\S]*?)```", t, re.IGNORECASE)
    if fence:
        t = fence.group(1).strip()
    return t.strip().rstrip(";")


def _execute_sql(sql: str) -> tuple[bool, str]:
    """执行 SQL 并返回 (是否成功, 说明或结果摘要)。"""
    sql = _normalize_sql(sql)
    if not sql or sql == ";":
        return False, "SQL 为空或无法解析"
    head = sql.lstrip().upper()
    if head.startswith(("SELECT", "SHOW", "DESC", "DESCRIBE", "EXPLAIN", "WITH")):
        try:
            r = get_db_table_info_mysql.invoke({"sql_query": sql})
            return True, repr(r) if r is not None else "(无返回行)"
        except Exception as e:
            return False, str(e)
    return run_write_sql(sql)


class SQLReflectionPattern:
    """Write → Verify（真实执行 SQL）→ 失败则回到 Write，成功或达到最大次数则结束。

    增删改查均在 verify 中完成：查询走查询接口，INSERT/UPDATE/DELETE 走执行并提交。
    不再拆成「只写 SQL」与「另由 agent 执行」两步。"""

    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
        self._write_llm = ChatOpenAI(
            model="qwen-plus",
            temperature=0,
            streaming=False,
        ).with_structured_output(SQLDraft)

    def build_graph(self):
        g = StateGraph(SQLReflectionState)
        g.add_node("write", self._write_node)
        g.add_node("verify", self._verify_node)
        
        g.add_edge(START, "write")
        g.add_edge("write", "verify")
        g.add_conditional_edges(
            "verify",
            self._route_after_verify,
            {"continue": "write", "end": END},
        )
        return g.compile()

    def _route_after_verify(self, state: SQLReflectionState) -> Literal["continue", "end"]:
        if state["execution_ok"]:
            return "end"
        if state["iteration"] >= self.max_iterations:
            return "end"
        return "continue"

    def _write_node(self, state: SQLReflectionState):
        task = state["task"]
        schema = state["table_schema"]
        prev_sql = (state.get("draft") or "").strip()
        err = (state.get("execution_message") or "").strip()
        it = int(state.get("iteration") or 0)

        if it == 0:
            human = (
                f"【任务】\n{task}\n\n【表结构】\n{schema}\n\n"
                "请输出一条满足任务的 SQL。"
            )
        else:
            human = (
                f"【任务】\n{task}\n\n【表结构】\n{schema}\n\n"
                f"【上一轮 SQL】\n{prev_sql}\n\n"
                f"【执行反馈】\n{err}\n\n"
                "请根据反馈写出修正后的完整 SQL（整句替换）。"
            )

        draft_obj = self._write_llm.invoke(
            [
                SystemMessage(content=SQL_REFLECTION_WRITE_SYSTEM),
                HumanMessage(content=human),
            ]
        )

        if not isinstance(draft_obj, SQLDraft):
            draft_obj = SQLDraft(sql=str(draft_obj))
        
        sql = (draft_obj.sql or "").strip()
        round_no = it + 1

        return {
            "draft": sql,
            "messages": [AIMessage(content=f"[第 {round_no} 轮编写]\n{sql}")],
        }

    def _verify_node(self, state: SQLReflectionState):
        ok, msg = _execute_sql(state.get("draft") or "")
        new_iter = int(state.get("iteration") or 0) + 1
        summary = f"[第 {new_iter} 次验证] {'成功' if ok else '失败'}: {msg}"
        return {
            "execution_ok": ok,
            "execution_message": msg,
            "iteration": new_iter,
            "messages": [AIMessage(content=summary)],
        }


@lru_cache(maxsize=1)
def _sql_reflection_graph():
    return SQLReflectionPattern(max_iterations=3).build_graph()


def _invoke_sql_reflection(task: str, table_schema: str) -> dict:
    return _sql_reflection_graph().invoke(
        {
            "task": task,
            "table_schema": table_schema,
            "draft": "",
            "execution_ok": False,
            "execution_message": "",
            "iteration": 0,
            "messages": [HumanMessage(content=task)],
        }
    )


def _load_table_schemas(table_name: str) -> tuple[str, str]:
    """支持单表或多表：英文/中文逗号分隔，返回 (合并后的结构文本, 表名列表展示串)。"""
    raw = (table_name or "").replace("，", ",")
    names = [t.strip() for t in raw.split(",") if t.strip()]
    if not names:
        return "", "(未提供有效表名)"
    blocks: list[str] = []
    for name in names:
        schema = get_db_table_structure_mysql.invoke({"table_name": name})
        blocks.append(f"【表 `{name}` 结构】\n{schema!s}")
    return "\n\n".join(blocks), ", ".join(names)


@tool
def call_sql_agent(query: str, table_name: str):
    """
    根据任务与表结构生成 SQL，并在反思流程内真实执行以验证（增删改查一体）。

    查询会返回结果摘要；INSERT/UPDATE/DELETE 会实际提交。失败时自动多轮修正 SQL。

    参数：
        query: 需要完成的任务描述
        table_name: 涉及的一个或多个表名；多表时用英文逗号分隔（如 appointments,therapists），
            会分别拉取每张表的 DESC，以便正确使用各表字段名（如技师主键、状态字段等）。
    返回：
        最后一版 SQL、是否验证成功、执行结果或错误说明
    """
    print(f"agent1需要做什么：{query}, 根据表格：{table_name}")
    schema_text, names_display = _load_table_schemas(table_name)
    final = _invoke_sql_reflection(
        query + "\n（涉及表：" + names_display + "）",
        schema_text,
    )
    sql = final.get("draft") or ""
    ok = bool(final.get("execution_ok"))
    msg = final.get("execution_message") or ""
    out = f"验证通过: {ok}\nSQL:\n{sql}\n\n执行说明:\n{msg}"
    print(f"agent1返回的结果：{out}")
    return out