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

# local에 있는 api key 읽기
def read_env():
    with open("../.env", "r") as f:
        return f.read().strip()
    
openai.api_key = read_env()
os.environ["OPENAI_API_KEY"] = openai.api_key

# 데이터 읽기
file = open("data.txt", "r")
content = file.read()
lines = content.strip().split("\n")
data_dict = {}
for idx,  line in enumerate(lines):
    if idx == 0  or idx == 1 or line == "":
        continue
    if line.startswith("#"):
        key = line[1:]  # '#' 문자를 제외한 나머지 문자열을 key로 사용
        data_dict[key] = ""
    else:
        data_dict[key] += line.strip()  # key에 해당하는 value에 문자열 추가

async def callback_handler(request: ChatbotRequest) -> dict:

    # ===================== start =================================
    llm = ChatOpenAI(temperature=0.1)
    search = DuckDuckGoSearchAPIWrapper()
    search.region = 'kr-kr'

    # 유저 질문과 가장 연관도가 높은 key값을 가져오기
    main_title = "카카오싱크"
    query = request.userRequest.utterance
    keyword_system_message = f"""
        assistant는 user의 질문이 아래 {len(data_dict.keys())}개 값중에서 가장 연관 높은 것을 뽑아줘.\n
        {data_dict.keys()}\n
        format은 바꾸지말고 그대로 return해줘.
        """
    keyword_system_message_prompt = SystemMessage(content=keyword_system_message)
    keyword_human_template = "{query}"
    keyword_human_message_prompt = HumanMessagePromptTemplate.from_template(keyword_human_template)
    chat_prompt = ChatPromptTemplate.from_messages([keyword_system_message_prompt, keyword_human_message_prompt])
    
    chain = LLMChain(llm=llm, prompt=chat_prompt, verbose=True)
    keyword = chain.run(query=query)
    print(f"keyword : {keyword}")

    # data_dict keyword가 있으면 해당 value를 가져오고 없으면 검색해서 가져오기
    if keyword in data_dict.keys():
        text = data_dict[keyword]
    else:
        search_results = search.results(main_title + ' ' + keyword, max_results=3)
        text = {}
        for search_result in search_results:
            text[search_result['title']] = search_result['link']

    system_message = f"assistant는 user의 질문에 대한 답변을 합니다."
    system_message_prompt = SystemMessage(content=system_message)
    human_template = "{query}에 대한 답변을\n\n {text}\n\n 위 내용을 참고해서 200자 이내로 답장해줘"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    
    chain = LLMChain(llm=llm, prompt=chat_prompt, verbose=True)
    output_text = chain.run(query=query, text=text)

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