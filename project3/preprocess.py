import numpy as np
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import openai
import os

def preprocess(path="./kakao_data/sink.txt"):
    with open(path, "r") as f:
            lines = f.readlines()
    output = []
    current_paragraph = ""
    for line in lines:
        if line.startswith('#'):
            if current_paragraph:
                output.append(current_paragraph.strip().replace('#',''))
            current_paragraph = line.strip()
        else:
            current_paragraph += ' ' + line.strip()
    if current_paragraph:
        output.append(current_paragraph.strip()[:])

    ret = [output[0] + doc for doc in output[1:]]
    return ret

if __name__ == "__main__":
    preprocessed_sink = preprocess("./kakao_data/sink.txt")
    preprocessed_channel = preprocess("./kakao_data/channel.txt")
    preprocessed_social = preprocess("./kakao_data/social.txt")

    all_doc = preprocessed_sink + preprocessed_channel + preprocessed_social

    chunk_size = np.median([len(doc) for doc in all_doc])
    CHROMA_PERSIST_DIR = "../chroma-persist"
    CHROMA_COLLECTION_NAME = "dosu-bot"

    # local에 있는 api key 읽기
    def read_env():
        with open("../.env", "r") as f:
            return f.read().strip()
        
    openai.api_key = read_env()
    os.environ["OPENAI_API_KEY"] = openai.api_key

    for doc in all_doc:
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
        splited_doc = text_splitter.split_text(doc)

    Chroma.from_texts(
        splited_doc,
        OpenAIEmbeddings(),
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print('db success')