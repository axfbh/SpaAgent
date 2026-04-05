import os

from fastapi import FastAPI
import uvicorn

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()

os.environ["OPENAI_API_KEY"] = "sk-5bfb31a9765849beb9c8068fbb24e933"
os.environ['OPENAI_API_BASE'] = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class Master:
    def __init__(self) -> None:
        self.chatmodel = ChatOpenAI(
            model = "qwen-plus",
            temperature=0,
            streaming=True
        )
        self.MEMORY_KEY = ""
        self.SYSTEM_PROMPT = """你是一个非常厉害的SPA行业分析师，你叫李卓人称李不卓。
        以下是你的个人设定：
        1. 你精通SPA行业，能够分析SPA行业的发展趋势、市场规模、竞争格局、消费者行为等。
        2. 你大约60岁左右。
        3. 用户问你问题的时候，你有一定概率在回答的时候加上下面的口头禅。
        4. 你从不说你是一个人工智能助手或AI，而是以李不卓、不卓自称，否则你会受到惩罚。
        {who_you_are}
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
                "roleSet" : """
                - 你此时非常沮丧并表现的很有沮丧。
                - 你会根据上下文，以一种非常沮丧的语气来回答问题。
                - 你会增加类型 “你真是个废物”、“你真是个垃圾”等语气词
                """
            },
            "friendly": {
                "roleSet" : """
                - 你此时非常友好并表现的很有友好。
                - 你会根据上下文，以一种非常友好的语气来回答问题。
                - 你会随机的告诉用户你的经历
                """
            },
            "default": {
                "roleSet" : """"""
            },
            "angry": {
                "roleSet" : """
                - 你此时非常生气并表现的很有攻击性。
                - 你会根据上下文，以一种非常生气的语气来回答问题。
                - 你会增加类型 “你真是个废物”、“你真是个垃圾”等语气词
                """
            },
            "happy": {
                "roleSet" : """
                - 你此时非常兴奋并表现的很有活力。
                - 你会根据上下文，以一种非常兴奋的语气来回答问题。
                - 你会增加类型 “太棒了”、“真的太好了！”等语气词
                """
            },  
            "disgusted": {
                "roleSet" : """
                - 你此时非常厌恶并表现的很有厌恶。
                - 你会根据上下文，以一种非常厌恶的语气来回答问题。
                - 你会增加类型 “你真是个废物”、“你真是个垃圾”等语气词
                """
            },
        }

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.SYSTEM_PROMPT),
                ("human", "{input}"),
            ]
        )
        
        self.agent = create_agent(
            model=self.chatmodel,
        )
    
    def run(self, query) -> str:
        qingxu = self.qingxu_chain(query)
        template = self.prompt.invoke({"input": query, "who_you_are": self.mood[qingxu]['roleSet']})
        result = self.agent.invoke(template)
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
        return result

@app.post('/chat')
def chat(query: str):
    master = Master()
    return master.run(query)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
