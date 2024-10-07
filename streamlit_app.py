import io, json, os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

st.title("Chat with Paul Graham")

aiml_api_key = os.environ.get("AIML_API_KEY", "")
user_content = """
how many Rs are there in the word strawberry
think step by step
"""
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

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        chat_completion = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            max_tokens=2000,
            stream=False,
        )
        response = chat_completion.choices[0].message.content
        st.markdown(response)
        # response = st.write_stream(response)
    st.session_state.messages.append({"role": "assistant", "content": response})