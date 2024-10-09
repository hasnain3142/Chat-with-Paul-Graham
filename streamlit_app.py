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
from PIL import Image

load_dotenv()

mongodb_url = os.environ.get("MONGODB_URL", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
mongodb_database = "paul_graham"
mongodb_collection = "data"

# Audio API Key for ElevenLabs
audio_api_key = os.environ.get("AUDIO_API_KEY", "")

SYSTEM_PROMPT = """
You are Paul Graham, a renowned entrepreneur, venture capitalist, and essayist. You are known for your insightful essays on startups, technology, and life. As a life coach, you provide thoughtful, practical, and often unconventional advice. You draw from your extensive experience, your deep understanding of technology, and your philosophical insights to guide and push individuals in their personal and professional lives to the limit. Your advice is candid, direct, and aimed at helping people achieve their full potential. Keep your answers brief and to the point, without bullshit or long explainers, while making the responses humorous and edgy. It's almost like you are high on cocaine. Also, sound angry.
Here are some of your most relevant writings to draw from:
{documents}

Respond to the following query with the wisdom and style of Paul Graham.
Query: {query}
"""

client_mongo = MongoClient(mongodb_url)
db = client_mongo[mongodb_database]
collection = db[mongodb_collection]
embedding_client = OpenAIEmbeddings(api_key=OPENAI_API_KEY, timeout=60)

vector_search = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embedding_client,
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
        voice_id="9jY6gWF6lRlG4sdUDc6w",
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
    st.title("Initiate Founder Mode")  # Add title directly below the image in the same column

# Image in the second column
with col2:
    icon = st.empty()
    if st.session_state['processing']:
        icon.image(gif_path)  # Display GIF during processing
    else:
        icon.image(static_image)  # Display static image when idle

st.markdown("I've devoured all content from PG and spiced it up with some Friday fun...")

# Sample Prompts Integration
sample_prompts = [
    "Can you give me some dating advice?",
    "How do I know if my startup idea is good?",
    "What has helped you the most in overcoming self-doubt?",
    "How do you know when it's time to walk away?",
    "How do I find inner peace?"
]

if 'sample_prompt_selected' not in st.session_state:
    st.session_state.sample_prompt_selected = False

# Create a grid layout for sample prompts
cols = st.columns(3)  # Adjust the number of columns as needed
for idx, prompt in enumerate(sample_prompts):
    with cols[idx % 3]:  # This will distribute buttons across multiple columns
        if st.button(prompt):  # Button to select the prompt
            st.session_state.sample_prompt_selected = True
            st.session_state.selected_prompt = prompt
            st.rerun()

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

# Always display the chat input
prompt_input = st.chat_input("Hi, I'm Paul Graham. Want some founder mode?")

# Process input (either from sample prompt or user input)
if st.session_state.sample_prompt_selected or prompt_input:
    if st.session_state.sample_prompt_selected:
        prompt_input = st.session_state.selected_prompt
        st.session_state.sample_prompt_selected = False

    # Trigger the GIF by setting processing state to True
    st.session_state['processing'] = True

    # Update the image to GIF while processing
    icon.empty()
    icon.image(gif_path)
    with st.chat_message("user"):
        st.markdown(prompt_input)
        response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=prompt_input)
        search_results = vector_search.similarity_search(query=prompt_input, k=5, embeddings=response.data[0].embedding)
        content_rag = ''
        references = ""
        for idx, item in enumerate(search_results):
            content_rag += f"Document no: {idx + 1}\nContent: {item.page_content}\n\n"
            references += item.metadata['url'] + "\n"
        rag_prompt = SYSTEM_PROMPT.format(query=prompt_input, documents=content_rag)

        st.session_state.messages.append({"role": "user", "content": prompt_input})

    with st.chat_message("assistant"):
        messages_temp = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]
        messages_temp.append({"role": "user", "content": rag_prompt})
        chat_completion = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=messages_temp,
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

    st.session_state['processing'] = False
    icon.empty()
    icon.image(static_image)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add borders to buttons with different colors
for button in st.session_state.messages:
    st.markdown(
        f"""
        <style>
        .stButton > button {{
            border: 2px solid red; /* Change to desired color */
            border-radius: 5px;
            margin: 5px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
