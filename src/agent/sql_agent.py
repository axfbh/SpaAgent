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

【核心业务：同一技师重叠时段不可预约】
- 含义：同一 `therapist_id` 在同一天内，若已有一条未取消的预约占用时间段 A，则不能再接受时间段 B，只要 A 与 B 在时间上「有交集」就必须拒绝新单。
- 时间用「开始时刻 + 持续分钟数 duration」表示一条预约占用的区间。区间按半开区间理解：`[开始, 开始 + INTERVAL duration MINUTE)`；**上一单结束时刻恰好等于下一单开始时刻**不算重叠（例如旧单 12:00–13:00、新单 13:00–14:30 允许）。
- 实现方式：在 `INSERT INTO appointments ... SELECT ... FROM therapists ...` 上增加 `AND NOT EXISTS (...)`：子查询在 `appointments` 中查找「与新预约冲突」的已有行；若存在任何一行满足重叠条件，则整个 SELECT 不产生行，插入失败（这是正确的业务结果）。
- 必须在 NOT EXISTS 子查询里排除已取消预约：例如 `COALESCE(a.status,'') NOT IN ('已取消')`（列名以表结构为准）。
- 时长列名必须以 `DESC appointments` 为准（如 `duration`、`duration_minutes`），**禁止臆造列名**；若仅有开始时间、无时长列，则至少禁止「同一 therapist_id + 同一 appointment_date + 同一 appointment_time」的重复插入。

【重叠的判定公式（两段区间）】
记 新起 = `TIMESTAMP(新日期, 新时间)`，新止 = `DATE_ADD(新起, INTERVAL 新时长 MINUTE)`；
对已有行 `a`：旧起 = `TIMESTAMP(a.appointment_date, a.appointment_time)`，旧止 = `DATE_ADD(旧起, INTERVAL a.<时长列> MINUTE)`。
两区间重叠当且仅当：**新起 < 旧止 且 旧起 < 新止**。把该条件完整写进 NOT EXISTS 的 WHERE 中（对 `a` 与固定的新预约常量比较），不要用「只比较一个端点」的简化写法以免漏判。

【NOT EXISTS 骨架示例（占位符须替换为任务中的日期、时间、时长及真实列名）】
`AND NOT EXISTS ( SELECT 1 FROM appointments a WHERE a.therapist_id = <与插入行相同的技师ID> AND a.appointment_date = <新预约日期> AND COALESCE(a.status,'') NOT IN ('已取消') AND TIMESTAMP(<新日期>, <新时间>) < DATE_ADD(TIMESTAMP(a.appointment_date, a.appointment_time), INTERVAL a.<时长列> MINUTE) AND TIMESTAMP(a.appointment_date, a.appointment_time) < DATE_ADD(TIMESTAMP(<新日期>, <新时间>), INTERVAL <新时长> MINUTE) )`

【易错点】
- `therapists` 表里匹配技师编号用主键列（表结构常见为 `id`）；`appointments` 里外键列才是 `therapist_id`。不要在 `FROM therapists` 上误写 `therapist_id = ?`。
- 新预约的时长若任务给出（如 90 分钟），须与 INSERT 列表中的 duration 常量一致，并用于 DATE_ADD 计算新止。

其它规则：
1. 仅使用表结构中存在的库表与字段名。
2. 符合 MySQL 语法；需要时用反引号包裹标识符。
3. 下一条「验证」步骤会真实执行该语句：SELECT/SHOW 等走查询；INSERT/UPDATE/DELETE 会写入数据库并提交；INSERT 未插入行或重叠校验失败均会判失败。
4. 任务要求「仅在职可预约」时：用 `INSERT INTO ... SELECT ... FROM therapists t WHERE t.id = ? AND t.status = '在职'`（主键列名以表结构为准）。
5. 若上一轮 SQL 执行失败，必须根据错误信息修正后给出全新 SQL，不要重复无效语句。
6. 禁止 DROP DATABASE、TRUNCATE 整库等破坏性操作（除非任务明确要求且你确认表名无误）。
7. INSERT 受影响行数为 0 时，说明业务上预约未成立：须在执行说明中体现可能原因（技师非在职、时段与已有预约重叠、NOT EXISTS 阻止插入等），
   便于上层向顾客如实说明「为什么上不了钟」，而不是虚构成功或虚构可约时间。
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
        最后一版 SQL、是否验证成功、执行结果或错误说明。
        预约类写入失败时，执行说明中含未插入原因，应答侧应据此向顾客解释，不得编造可约时间。
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