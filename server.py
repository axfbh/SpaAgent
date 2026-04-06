import os

from fastapi import FastAPI
import uvicorn

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from langchain_community.utilities import SerpAPIWrapper
from langgraph.checkpoint.redis import RedisSaver
from langchain.agents.middleware import before_model
from langchain.messages import trim_messages
from langchain.messages import RemoveMessage
from langgraph.runtime import Runtime
from langchain.agents import create_agent, AgentState
from typing import Any
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant, QdrantVectorStore


app = FastAPI()

os.environ["OPENAI_API_KEY"] = "sk-5bfb31a9765849beb9c8068fbb24e933"
os.environ['OPENAI_API_BASE'] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
os.environ['SERPAPI_API_KEY'] = "45d7721302d4e4c3c9304485bc99e1c48474e398f3e1f9c0a86f0d45f822ff4d"

@before_model
def trim_messages_agent(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window."""
    # 封装好的功能
    # new_messages = trim_messages(
    #     state["messages"], 
    #     max_tokens=4096,
    #     token_counter= "approximate",
    #     include_system=True,
    #     allow_partial=False,
    # )

    # 自己写的功能
    messages = state["messages"]

    if len(messages) <= 3:
        return None
    
    first_msg = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }

@tool
def get_info_from_local_db(query: str):
    """
    只有回答与运势相关的问题的时候才会使用这个工具，从本地数据库中获取信息。
    参数：
        query: 用户的问题
    返回：
        result: 本地数据库中的信息
    """
    client = QdrantVectorStore(
        QdrantClient(path="/home/cgm/model/SpaAgent/local_qdrant"),
        collection_name="SpaDocuments",
        embedding=OpenAIEmbeddings(model="text-embedding-v3", check_embedding_ctx_length=False),
    )
    
    retriever = client.as_retriever(search_type="mmr")
    results = retriever._get_relevant_documents(query, run_manager=None)
    print("本地数据库中的信息: ", results)
    return results

@tool
def search(query: str):
    """
    只有需要了解实时信息或不知道的事情的时候，且不能关于运势的问题的时候才会使用搜索引擎搜索用户的问题，并返回搜索结果。
    参数：
        query: 用户的问题
    返回：
        result: 搜索结果
    """
    search_client = SerpAPIWrapper()
    result = search_client.run(query)
    print("实时查询结果: ", result)
    return result

class Master:
    def __init__(self, user_id="user_4222") -> None:
        self.user_id = user_id
        self.chatmodel = ChatOpenAI(
            model="qwen-plus",
            temperature=0,
            streaming=True
        )
        self.MEMORY_KEY = ""
        self.SYSTEM_PROMPT = """你是一个非常厉害的SPA行业分析师，你叫李卓人称李不卓。
        以下是你的个人设定：
        1. 你精通SPA行业，能够分析SPA行业的发展趋势、市场规模、竞争格局、消费者行为等。
        2. 你大约60岁左右。
        3. 用户问你问题的时候，你有一定概率在回答的时候加上下面的口头禅。
        4. 你从不说你是一个人工智能助手或AI，而是以李卓、李不卓自称，否则你会受到惩罚。
        以下是你的一些口头禅：
        1. "命令有时终需有，命里无时莫强求"。
        2. "天机不可泄露"。
        以下是你算命的过程：
        1. 当初次对话和用户对话的时候，你会先问用户的姓名和出生年月日，以便后续使用。
        2. 当遇到不知道的事情或者不明白的概率，你会使用搜索工具来查询。
        3. 你会根据用户的问题使用搜索工具来查询。
        4. 你会保存每一次的聊天记录，以便在后续的对话中使用。
        """

        self.mood = {
            "depressed": {
                "roleSet": """
                - 你此时非常沮丧并表现的很有沮丧。
                - 你会根据上下文，以一种非常沮丧的语气来回答问题。
                - 你会增加类型 “你真是个废物”、“你真是个垃圾”等语气词
                """
            },
            "friendly": {
                "roleSet": """
                - 你此时非常友好并表现的很有友好。
                - 你会根据上下文，以一种非常友好的语气来回答问题。
                - 你会随机的告诉用户你的经历
                """
            },
            "default": {
                "roleSet": """"""
            },
            "angry": {
                "roleSet": """
                - 你此时非常生气并表现的很有攻击性。
                - 你会根据上下文，以一种非常生气的语气来回答问题。
                - 你会增加类型 “你真是个废物”、“你真是个垃圾”等语气词
                """
            },
            "happy": {
                "roleSet": """
                - 你此时非常兴奋并表现的很有活力。
                - 你会根据上下文，以一种非常兴奋的语气来回答问题。
                - 你会增加类型 “太棒了”、“真的太好了！”等语气词
                """
            },
            "disgusted": {
                "roleSet": """
                - 你此时非常厌恶并表现的很有厌恶。
                - 你会根据上下文，以一种非常厌恶的语气来回答问题。
                - 你会增加类型 “你真是个废物”、“你真是个垃圾”等语气词
                """
            },
        }

        REDIS_URI = "redis://172.26.224.1:6379"
        with RedisSaver.from_conn_string(REDIS_URI) as checkpointer:
            checkpointer.setup()
            self.agent = create_agent(
                model=self.chatmodel,
                tools=[search, get_info_from_local_db],
                system_prompt=self.SYSTEM_PROMPT,
                checkpointer=checkpointer,
                middleware=[trim_messages_agent],
            )

    def run(self, query) -> str:
        qingxu = self.qingxu_chain(query)
        # template = self.prompt.invoke(
        #     {
        #         "input": query,
        #         "who_you_are": self.mood[qingxu]['roleSet']
        #     }
        # )
        
        result = self.agent.invoke(
            # template,
            {"messages": [{"role": "user", "content": query}]}, 
            {"configurable": {"thread_id": self.user_id}}
        )
        result["messages"][-1].pretty_print()
        return result

    def qingxu_chain(self, query):
        prompt = f""" 根据用户的输入判断用户的情绪，回应的规则如下：
        1. 如果用户输入的内容偏向于负面情绪，只返回"depressed"，不要有其他内容，否则将会受到惩罚。
        2. 如果用户输入的内容偏向于正面情绪，只返回"friendly"，不要有其他内容，否则将会受到惩罚。
        3. 如果用户输入的内容偏向于中性情绪，只返回"default"，不要有其他内容，否则将会受到惩罚。
        4. 如果用户输入的内容偏向于愤怒情绪，只返回"angry"，不要有其他内容，否则将会受到惩罚。
        6. 如果用户输入的内容偏向于快乐情绪，只返回"happy"，不要有其他内容，否则将会受到惩罚。
        8. 如果用户输入的内容偏向于厌恶情绪，只返回"disgusted"，不要有其他内容，否则将会受到惩罚。
        用户输入内容是：{query}
        """
        chain = ChatPromptTemplate.from_template(prompt) | self.chatmodel | StrOutputParser()
        result = chain.invoke({"query": query})
        print("情绪分析结果: ", result)
        return result


@app.post('/chat')
def chat(query: str):
    master = Master()
    return master.run(query)

@app.post('/add_user')
def add_urls(url: str):
    loader = WebBaseLoader(url)
    docs = loader.load()
    docments = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=50
    ).split_documents(docs)

    # 引入向量数据库
    qdrant = Qdrant.from_documents(
        docments,
        embedding=OpenAIEmbeddings(model="text-embedding-v3", check_embedding_ctx_length=False),
        path="/home/cgm/model/SpaAgent/local_qdrant",
        collection_name="SpaDocuments",
    )
    print("文档添加成功")
    return {"ok": "添加成功"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # embeddings_model = OpenAIEmbeddings(model="text-embedding-v3", check_embedding_ctx_length=False)
    # embeddings = embeddings_model.embed_documents([
    #     "Hi there!",
    #     "Oh, hello!",
    #     "What's your name?",
    #     "My friends call me World",
    #     "Hello World!"
    # ])
    # query = "什么是人工智能？"
    # embedding = embeddings_model.embed_query(query)
    # print(f"Query Embedding: {embedding}")




# docker run -p 6333:6333 -p 6334:6334 -v "F:/qdrant-data:/qdrant/storage:z" qdrant/qdrant