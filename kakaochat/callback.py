from dto import ChatbotRequest
from samples import list_card
import aiohttp
import time
import logging
import openai
import os
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage

def read_env():
    with open("../.env", "r") as f:
        return f.read().strip()
    
openai.api_key = read_env()
os.environ["OPENAI_API_KEY"] = openai.api_key

file = open("data.txt", "r")
content = file.read()
lines = content.strip().split("\n")  # 개행 문자로 문자열을 분리하여 리스트로 변환
result = {}

for idx,  line in enumerate(lines):
    if idx == 0  or idx == 1 or line == "":
        continue
    if line.startswith("#"):
        key = line[1:]  # '#' 문자를 제외한 나머지 문자열을 key로 사용
        result[key] = ""
    else:
        result[key] += line.strip()  # key에 해당하는 value에 문자열 추가

async def callback_handler(request: ChatbotRequest) -> dict:

    # ===================== start =================================
    title = "카카오싱크"
    query = request.userRequest.utterance
    system_message = f"assistant는 user의 질문이 {result.keys()}중에서 가장 연관도가 높은 것을 뽑아줘. format은 바꾸지말고 그대로 return해줘."
    system_message_prompt = SystemMessage(content=system_message)
    human_template = "{query}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,
                                                    human_message_prompt])
    
    llm = ChatOpenAI(temperature=0.1)
    chain = LLMChain(llm=llm, prompt=chat_prompt)

    output_text = chain.run(query=query)

   # 참고링크 통해 payload 구조 확인 가능
    payload = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": output_text
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