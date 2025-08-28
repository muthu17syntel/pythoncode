# ---------- IMPORTS ----------
import os
import streamlit as st
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery

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
    # Step 1: Get embedding
    embedding = client.embeddings.create(
        model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        input=query
    )
    vector = embedding.data[0].embedding

    # Step 2: Vector search
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

    # Step 3: GPT answer
    completion = client.chat.completions.create(
        model=AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Use context to answer."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        max_tokens=500
    )
    return completion.choices[0].message.content

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Azure AI Search + GPT", layout="centered")

st.title("🔎 Automation Assets Bot")
st.write("This app uses automation documents and scripts to answer your queries")

# Input box
user_question = st.text_input("Enter your question:")

if st.button("Ask"):
    if user_question.strip():
        with st.spinner("Searching and generating answer..."):
            try:
                answer = ask_gpt(user_question)
                st.subheader("Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a question.")
