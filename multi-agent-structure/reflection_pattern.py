import os
from typing_extensions import TypedDict
from typing import Annotated

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages

os.environ["OPENAI_API_KEY"] = "sk-5bfb31a9765849beb9c8068fbb24e933"
os.environ["OPENAI_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class ReflectionState(TypedDict):
    """反思循环状态：任务说明、当前稿件、评审反馈与分数、已完成评审轮数。"""

    task: str
    draft: str
    feedback: str
    score: float
    iteration: int
    messages: Annotated[list, add_messages]


class ReviewResult(BaseModel):
    """评审结构化输出：1–10 分 + 可执行的修改建议。"""

    score: float = Field(ge=1, le=10, description="对当前稿件质量的打分，1 最低 10 最高")
    feedback: str = Field(description="指出主要问题并给出具体可执行的修改建议，便于下一版改写")


llm = ChatOpenAI(model="qwen-flash", temperature=0.3)
review_llm = llm.with_structured_output(ReviewResult)


WRITE_SYSTEM = """你是一名专业作者。请只输出「正文稿件」本身，不要输出标题如「修改稿」「初稿」等说明性前缀。
若提供了「上一版稿件」和「评审意见」，请在保持任务要求的前提下充分吸收意见完成改写；若只有任务说明，则写一版尽力完成的初稿。"""


REVIEW_SYSTEM = """你是一名严格且具体的编辑。请根据任务要求评估当前稿件，打出 1–10 的整数或一位小数分数。
打分参考：6 分以下表示有明显缺失或跑题；7–7.5 可用但平淡或有瑕疵；8 分以上表示内容扎实、结构清晰、基本可直接交付。
请给出简短、可执行的修改建议（不要空洞夸奖）。输出必须符合约定的结构化字段。"""


class ReflectionPattern:
    def __init__(self, max_iterations: int = 3, score_threshold: float = 8.0):
        self.max_iterations = max_iterations
        self.score_threshold = score_threshold

    def build_graph(self):
        graph = StateGraph(ReflectionState)

        graph.add_node("write", self._write_node)
        graph.add_node("review", self._review_node)

        graph.add_edge(START, "write")
        graph.add_edge("write", "review")
        graph.add_conditional_edges(
            "review",
            self._should_continue,
            {"continue": "write", "end": END},
        )
        return graph.compile()

    def _should_continue(self, state: ReflectionState) -> str:
        if state["score"] >= self.score_threshold:
            return "end"
        if state["iteration"] >= self.max_iterations:
            return "end"
        return "continue"

    def _write_node(self, state: ReflectionState):
        task = state["task"]
        prev = state.get("draft") or ""
        fb = state.get("feedback") or ""

        if not prev.strip():
            human = f"【写作任务】\n{task}\n\n请完成一版初稿。"
        else:
            human = (
                f"【写作任务】\n{task}\n\n"
                f"【上一版稿件】\n{prev}\n\n"
                f"【评审意见】\n{fb}\n\n"
                "请根据任务与评审意见写出改进后的完整新版本（整篇替换，不要只写补丁）。"
            )

        resp = llm.invoke(
            [
                SystemMessage(content=WRITE_SYSTEM),
                HumanMessage(content=human),
            ]
        )
        text = (resp.content or "").strip()
        return {
            "draft": text,
            "messages": [AIMessage(content=f"[第 {state['iteration'] + 1} 轮写作]\n{text}")],
        }

    def _review_node(self, state: ReflectionState):
        task = state["task"]
        draft = state["draft"]
        human = f"【任务】\n{task}\n\n【待评审稿件】\n{draft}"

        result = review_llm.invoke(
            [
                SystemMessage(content=REVIEW_SYSTEM),
                HumanMessage(content=human),
            ]
        )
        if not isinstance(result, ReviewResult):
            result = ReviewResult(score=6.0, feedback=str(result))

        new_iter = state["iteration"] + 1
        summary = f"[第 {new_iter} 轮评审] 分数 {result.score}/10。意见：{result.feedback}"
        return {
            "score": result.score,
            "feedback": result.feedback,
            "iteration": new_iter,
            "messages": [AIMessage(content=summary)],
        }


def stream_graph_updates(graph, user_input: str):
    init: ReflectionState = {
        "task": user_input,
        "draft": "",
        "feedback": "",
        "score": 0.0,
        "iteration": 0,
        "messages": [HumanMessage(content=user_input)],
    }
    for event in graph.stream(init):
        for node_name, value in event.items():
            if not isinstance(value, dict):
                continue
            draft = value.get("draft")
            score = value.get("score")
            it = value.get("iteration")
            fb = value.get("feedback")
            if draft:
                print(f"\n--- [{node_name}] 当前稿件（节选） ---\n{draft[:800]}{'…' if len(draft) > 800 else ''}\n")
            if score is not None and node_name == "review":
                print(f"--- [{node_name}] 迭代 {it} | 分数 {score}/10 ---")
                if fb:
                    print(f"评审意见：{fb}\n")


if __name__ == "__main__":
    reflection_pattern = ReflectionPattern(max_iterations=3, score_threshold=8.0)
    graph = reflection_pattern.build_graph()
    user_input = input("User: ")
    stream_graph_updates(graph, user_input)
