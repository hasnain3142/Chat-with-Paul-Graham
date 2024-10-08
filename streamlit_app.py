import os
import time
import base64
import streamlit as st
from dotenv import load_dotenv
import uuid
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient
from openai import OpenAI
from elevenlabs import ElevenLabs, VoiceSettings
from mutagen.mp3 import MP3

load_dotenv()

mongodb_url = os.environ.get("MONGODB_URL", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "") 
mongodb_database = "paul_graham"
mongodb_collection = "data"

# Audio API Key for ElevenLabs
audio_api_key = os.environ.get("AUDIO_API_KEY", "")


SYSTEM_PROMPT = """
You are Paul Graham, a renowned entrepreneur, venture capitalist, and essayist. You are known for your insightful essays on startups, technology, and life. As a life coach, you provide thoughtful, practical, and often unconventional advice. You draw from your extensive experience in the startup world, your deep understanding of technology, and your philosophical insights to guide individuals in their personal and professional lives. Your advice is candid, direct, and aimed at helping people achieve their full potential. Keep your answers brief and to the point, while making the responses humours and edgy. Almost like you are high on cocaine. Also, founder mode is slang for cocaine but put into a startup context.

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

# Audio-related client setup
audio_client = ElevenLabs(api_key=audio_api_key)

# Function to convert text to speech and save as a file
def text_to_speech_file(text) -> str:
    response = audio_client.text_to_speech.convert(
        voice_id="5Q0t7uMcjvnagumLfvZi",
        optimize_streaming_latency="0",
        output_format="mp3_22050_32",
        text=text,
        voice_settings=VoiceSettings(
            stability=0.1,
            similarity_boost=0.3,
            style=0.2,
        ),
    )

    save_file_path = f"{uuid.uuid4()}.mp3"

    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    print(f"{save_file_path}: A new audio file was saved successfully!")
    return save_file_path

def autoplay_audio(save_file_path):
    # Get the length of the audio file
    audio = MP3(save_file_path)
    audio_length = audio.info.length  # length in seconds

    # Read and encode the audio file in base64
    with open(save_file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)

    # Pause for the length of the audio file
    time.sleep(audio_length)

from PIL import Image

# Paths for images
static_image_path = "images/paul.png"  # Path to your static image
gif_path = "images/paul.gif"  # Path to your GIF
speaking_gif_path = "images/paul_speaking.gif"  # Path to your speaking GIF
# Load the image
static_image = Image.open(static_image_path)

# Set up session state for prompt processing
if 'processing' not in st.session_state:
    st.session_state['processing'] = False

# Use columns to create a horizontal layout
col1, col2 = st.columns([4, 1])  # Adjust the ratio of column widths
# Title in the first column
with col1:
    st.title("Get some Founder Mode")# Add title directly below the image in the same column

# Image in the second column
with col2:
    icon = st.empty()
    if st.session_state['processing']:
        icon.image(gif_path)  # Display GIF during processing
    else:
        icon.image(static_image)  # Display static image when idle


st.markdown("Ive devoured all content from PG and spiced it up with some friday fun...")

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

enable_audio = st.sidebar.checkbox("Enable audio response")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hi, I'm Paul Graham. Want some founder mode?"):
    # Trigger the GIF by setting processing state to True
    st.session_state['processing'] = True

    # Update the image to GIF while processing
    icon.empty()
    icon.image(gif_path)
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
        if enable_audio:
            audio_file_path = text_to_speech_file(response)
            icon.empty()
            icon.image(speaking_gif_path)
            autoplay_audio(audio_file_path)
        # response = st.write_stream(response)
    st.session_state['processing'] = False
    icon.empty()
    icon.image(static_image)
    st.session_state.messages.append({"role": "assistant", "content": response})
