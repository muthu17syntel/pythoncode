import os
import streamlit as st
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
from streamlit_lottie import st_lottie
import requests

# ---------- Page Config ----------
st.set_page_config(page_title="AI Knowledge Assistant",
                   page_icon="🤖",
                   layout="wide")

# ---------- Custom CSS ----------
st.markdown("""
    <style>
        /* Background gradient */
        .stApp {
            background: linear-gradient(135deg, #1f1c2c, #928dab);
            color: white;
        }
        /* Chat bubbles */
        .user-bubble {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border-radius: 20px;
            margin: 5px;
            text-align: right;
            max-width: 70%;
        }
        .bot-bubble {
            background-color: #2c2c54;
            color: white;
            padding: 10px 15px;
            border-radius: 20px;
            margin: 5px;
            text-align: left;
            max-width: 70%;
        }
        /* Input box */
        textarea {
            border-radius: 10px !important;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Azure OpenAI Config ----------
AZURE_OPENAI_ENDPOINT = "https://muthuopenai.openai.azure.com"
AZURE_OPENAI_API_KEY = "28Jo9eIRdmoCCQTzUB0upgSIZpXksrfZnmhp2Vo46UQI7QrKqE18JQQJ99BGACYeBjFXJ3w3AAABACOGMV0T"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = "text-embedding-ada-002"
AZURE_OPENAI_CHAT_DEPLOYMENT = "gpt-35-turbo"

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2024-02-15-preview"
)

# ---------- Azure AI Search Config ----------
SEARCH_ENDPOINT = "https://testadosearch.search.windows.net"
SEARCH_KEY = "23kHJYzGON0i1pw9hb7Sx9R0hBzznwgCPkFwdZspPjAzSeDIkI11"
SEARCH_INDEX = "rag-1756292528423"

search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=SEARCH_INDEX,
    credential=AzureKeyCredential(SEARCH_KEY)
)

# ---------- Helper function ----------
def ask_gpt(query: str):
    embedding = client.embeddings.create(
        model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        input=query
    )
    vector = embedding.data[0].embedding

    vector_query = VectorizedQuery(
        vector=vector,
        k_nearest_neighbors=3,
        fields="text_vector",
        kind="vector"
    )

    results = search_client.search(
        search_text="",
        vector_queries=[vector_query],
        select=["chunk", "title"]
    )

    context = "\n".join([doc["chunk"] for doc in results])

    completion = client.chat.completions.create(
        model=AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Use context to answer."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        max_tokens=500
    )
    return completion.choices[0].message.content

# ---------- Lottie Animation Loader ----------
def load_lottieurl(url: str):
    try:
        r = requests.get(url, verify=False)  # 👈 bypass SSL check
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        return None


lottie_robot = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_p9y3zt0h.json")

# ---------- Sidebar ----------
with st.sidebar:
    st_lottie(lottie_robot, height=200, key="robot")
    st.title("⚡ AI Knowledge Bot")
    st.markdown("Ask me anything about your data 🚀")

# ---------- Main Chat UI ----------
if "history" not in st.session_state:
    st.session_state.history = []

st.title("🤖 Your Smart AI Assistant")
query = st.text_input("💬 Type your question and hit Enter", key="input")

if query:
    answer = ask_gpt(query)
    st.session_state.history.append({"role": "user", "content": query})
    st.session_state.history.append({"role": "bot", "content": answer})

# ---------- Display chat ----------
for chat in st.session_state.history:
    if chat["role"] == "user":
        st.markdown(f"<div class='user-bubble'>{chat['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-bubble'>{chat['content']}</div>", unsafe_allow_html=True)

