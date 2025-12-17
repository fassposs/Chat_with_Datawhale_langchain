from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

chat = ChatZhipuAI(
    model="glm-4.5-flash",
    temperature=0.5,
    zhipuai_api_key=".wyOnha8gBn8U9BoB",
    # zhipuai_api_key="sk-7itaucv9ZCYdQzy3MFThr5O9RLPMLC1ZVuOXC2h5Tu4F8SK7",
    # api_base="https://api.qingyuntop.top/v1",
)
messages = [
    AIMessage(content="Hi."),
    SystemMessage(content="Your role is a poet."),
    HumanMessage(content="简单介绍一下人工智能,不要太多文字"),
]

response = chat.invoke(messages)
print(response.content)  # Displays the AI-generated poem