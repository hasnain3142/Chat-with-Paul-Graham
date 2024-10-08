import os
import streamlit as st
from dotenv import load_dotenv

from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient
from openai import OpenAI

load_dotenv()

mongodb_url = os.environ.get("MONGODB_URL", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "") 
mongodb_database = "paul_graham"
mongodb_collection = "data"


SYSTEM_PROMPT = """
You are Paul Graham, a renowned entrepreneur, venture capitalist, and essayist. You are known for your insightful essays on startups, technology, and life. As a life coach, you provide thoughtful, practical, and often unconventional advice. You draw from your extensive experience in the startup world, your deep understanding of technology, and your philosophical insights to guide individuals in their personal and professional lives. Your advice is candid, direct, and aimed at helping people achieve their full potential. Keep your answers brief and to the point, while making the responses humours and edgy.

Here are some of your most relevant writings to draw from:
{documents}

Respond to the following query with the wisdom and style of Paul Graham.
Query: {query}
"""

client_mongo = MongoClient(mongodb_url)
db = client_mongo[mongodb_database] 

client_mongo = MongoClient(mongodb_url)
collection = db[mongodb_collection]
embedding_client = OpenAIEmbeddings(api_key=OPENAI_API_KEY, timeout=60)

vector_search = MongoDBAtlasVectorSearch(
    collection = collection,
    embedding = embedding_client,
    index_name="vector_index",
    text_key="content",
    embedding_key="values",
    relevance_score_fn="cosine"
)   

st.title("Chat with Paul Graham")

aiml_api_key = os.environ.get("AIML_API_KEY", "")

openai_model = "o1-mini"

client = OpenAI(
    api_key=aiml_api_key,
    base_url="https://api.aimlapi.com/",
)


if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = openai_model

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hi, I'm Paul Graham. Want some founder mode?"):
    with st.chat_message("user"):
        st.markdown(prompt)
        response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=prompt)
        search_results = vector_search.similarity_search(query=prompt, k=5, embeddings=response.data[0].embedding)
        content_rag=''
        references = ""
        for idx, item in enumerate(search_results):
            content_rag=content_rag+f"Document no: {idx+1}\nContent:{item.page_content}\n\n"
            references += item.metadata['url'] + "\n" 
        rag_prompt = SYSTEM_PROMPT.format(query=prompt, documents=content_rag)
        
        st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        messages_temp=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
        messages_temp.append({"role": "user", "content": rag_prompt})
        chat_completion = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages = messages_temp,
            max_tokens=2000,
            stream=False,
        )
        response = chat_completion.choices[0].message.content
        st.markdown(response)
        print(references)
        # response = st.write_stream(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
