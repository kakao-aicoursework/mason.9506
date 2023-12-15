from dto import ChatbotRequest
from samples import list_card
import aiohttp
import time
# import logging
import openai
import os
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from langchain.chains import ConversationChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory

CHROMA_PERSIST_DIR = "../chroma-persist"
CHROMA_COLLECTION_NAME = "dosu-bot"

# local에 있는 api key 읽기
def read_env():
    with open("../.env", "r") as f:
        return f.read().strip()
    
openai.api_key = read_env()
os.environ["OPENAI_API_KEY"] = openai.api_key

# template 일기
def read_prompt_template(file_path: str):
    with open(file_path, "r") as f:
        prompt_template = f.read()
    return prompt_template

# LLMChain 생성
def create_chain(llm, template_path, output_key):
    return LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(
            template=read_prompt_template(template_path)
        ),
        output_key=output_key,
        verbose=True,
    )

def get_chat_history(history):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="user_message",
        chat_memory=history,
    )
    return memory.buffer

# db 불러오기
db = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=OpenAIEmbeddings(),
    collection_name=CHROMA_COLLECTION_NAME,
)

# 초기 history 생성
history = FileChatMessageHistory("./chat_history.json")

async def callback_handler(request: ChatbotRequest) -> dict:

    # ===================== start =================================
    llm = ChatOpenAI(temperature=0.1)
    
    # search = DuckDuckGoSearchAPIWrapper()
    # search.region = 'kr-kr'
    
    # user message
    user_message = request.userRequest.utterance
    
    # LLMChain 생성
    llm = ChatOpenAI(temperature=0.1, max_tokens=1024, model="gpt-3.5-turbo")
    qna_chain = create_chain(
        llm=llm,
        template_path="./prompt/question_and_answer.txt",
        output_key="question_and_answer",
    )
    intent_chain = create_chain(
        llm=llm,
        template_path="./prompt/intent.txt",
        output_key="intent",
    )
    default_chain =  create_chain(
        llm=llm,
        template_path="./prompt/default.txt",
        output_key="default",
    )

    # 답변
    context = dict(user_message=user_message)
    context["input"] = context["user_message"]
    context["chat_history"] = get_chat_history(history)[-1000:]  # history 최근 1000자만 사용

    # user message 의도 파악
    intent = intent_chain.run(context)
    print(f"intent = {intent}")

    # 의도에 따른 모델 실행
    if intent == "question":
        context["related_documents"] = db.similarity_search(context["user_message"])[:3]
        answer = qna_chain.run(context)
    else:
        answer = default_chain.run(context)

    print(f"answer = {answer}")

    history.add_user_message(user_message)
    history.add_ai_message(answer)

   # 참고링크 통해 payload 구조 확인 가능
    payload = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": answer
                    }
                }
            ]
        }
    }
    # ===================== end =================================
    # 참고링크1 : https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/ai_chatbot_callback_guide
    # 참고링크1 : https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/answer_json_format

    time.sleep(1.0)

    url = request.userRequest.callbackUrl

    if url:
        async with aiohttp.ClientSession() as session:
            async with session.post(url=url, json=payload, ssl=False) as resp:
                await resp.json()