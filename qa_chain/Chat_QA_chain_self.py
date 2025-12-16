# from langchain_core.prompts import PromptTemplate
# from langchain_core import RetrievalQA
# from langchain_chroma import Chroma
# from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
# from langchain_chroma import ConversationBufferMemory
# from langchain.chat_models import ChatOpenAI
import sys
sys.path.append('/Users/lta/Desktop/llm-universe/project')
from qa_chain.model_to_llm import model_to_llm
from qa_chain.get_vectordb import get_vectordb
import re

class Chat_QA_chain_self:
    """"
    带历史记录的问答链  
    - model：调用的模型名称
    - temperature：温度系数，控制生成的随机性
    - top_k：返回检索的前k个相似文档
    - chat_history：历史记录，输入一个列表，默认是一个空列表
    - history_len：控制保留的最近 history_len 次对话
    - file_path：建库文件所在路径
    - persist_path：向量数据库持久化路径
    - appid：星火
    - api_key：星火、百度文心、OpenAI、智谱都需要传递的参数
    - Spark_api_secret：星火秘钥
    - Wenxin_secret_key：文心秘钥
    - embeddings：使用的embedding模型
    - embedding_key：使用的embedding模型的秘钥（智谱或者OpenAI）  
    """
    def __init__(self,model:str, temperature:float=0.0, top_k:int=4, chat_history:list=[], file_path:str=None, persist_path:str=None, appid:str=None, api_key:str=None, Spark_api_secret:str=None,Wenxin_secret_key:str=None, embedding = "openai",embedding_key:str=None):
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.chat_history = chat_history
        #self.history_len = history_len
        self.file_path = file_path
        self.persist_path = persist_path
        self.appid = appid
        self.api_key = api_key
        self.Spark_api_secret = Spark_api_secret
        self.Wenxin_secret_key = Wenxin_secret_key
        self.embedding = embedding
        self.embedding_key = embedding_key


        self.vectordb = get_vectordb(self.file_path, self.persist_path, self.embedding,self.embedding_key)
        
    
    def clear_history(self):
        "清空历史记录"
        return self.chat_history.clear()

    
    def change_history_length(self,history_len:int=1):
        """
        保存指定对话轮次的历史记录
        输入参数：
        - history_len ：控制保留的最近 history_len 次对话
        - chat_history：当前的历史对话记录
        输出：返回最近 history_len 次对话
        """
        n = len(self.chat_history)
        return self.chat_history[n-history_len:]

 
    def _combine_chat_history_and_question(self,inputs):
        """组合聊天历史和当前问题"""
        chat_history = inputs["chat_history"] or []
        question = inputs["question"]
        
        # 构造聊天历史字符串
        chat_history_str = ""
        for human, ai in chat_history:
            chat_history_str += f"Human: {human}\nAI: {ai}\n"
        
        return {
            "question": question,
            "chat_history": chat_history_str,
            "context": self.format_docs(inputs["context"])
        }

    def format_docs(self, docs):
        """格式化检索到的文档"""
        return "\n\n".join(doc.page_content for doc in docs)

    def answer(self, question:str=None,temperature = None, top_k = 4):
        """"
        核心方法，调用问答链
        arguments: 
        - question：用户提问
        """
        
        if len(question) == 0:
            return "", self.chat_history
        
        if len(question) == 0:
            return ""
        
        if temperature == None:
            temperature = self.temperature

        # 获取一个大模型  
        llm = model_to_llm(
            self.model, 
            temperature, 
            self.appid, 
            self.api_key, 
            self.Spark_api_secret,
            self.Wenxin_secret_key
        )

        #self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # 获取检索数据库
        retriever = self.vectordb.as_retriever(search_type="similarity",   
                                        search_kwargs={'k': top_k})  #默认similarity，k=4
        # ===========================================
        condense_question_system_template = (
            "请根据聊天记录总结用户最近的问题，"
            "如果没有多余的聊天记录则返回用户的问题。"
        )
        condense_question_prompt = ChatPromptTemplate([
                ("system", condense_question_system_template),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
            ])

        retrieve_docs = RunnableBranch(
            (lambda x: not x.get("chat_history", False), (lambda x: x["input"]) | retriever, ),
            condense_question_prompt | llm | StrOutputParser() | retriever,
        )

        system_prompt = (
            "你是一个问答任务的助手。 "
            "请使用检索到的上下文片段回答这个问题。 "
            "如果你不知道答案就说不知道。 "
            "请使用简洁的话语回答用户。"
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
            ]
        )
        qa_chain = (
            RunnablePassthrough().assign(context=combine_docs)
            | qa_prompt
            | llm
            | StrOutputParser()
        )

        qa = RunnablePassthrough().assign(
            context = retrieve_docs, 
            ).assign(answer=qa_chain)
        # ===========================================
        # qa = ConversationalRetrievalChain.from_llm(
        #     llm = llm,
        #     retriever = retriever
        # )
        
        #print(self.llm)
        result = qa({"question": question,"chat_history": self.chat_history})       #result里有question、chat_history、answer
        answer =  result['answer']
        answer = re.sub(r"\\n", '<br/>', answer)
        self.chat_history.append((question,answer)) #更新历史记录

        return self.chat_history  #返回本次回答和更新后的历史记录
        
















