# uvicorn gtp-using-ai-search:app --host 127.0.0.1 --port 8000
# gunicorn -w 1 -k uvicorn.workers.UvicornWorker gtp-using-ai-search:app  -- for production in app service.

# ---------- CLIENTS ----------
import os
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
from fastapi import FastAPI
from pydantic import BaseModel

# ---------- Azure OpenAI Config ----------
AZURE_OPENAI_ENDPOINT = "https://muthuopenai.openai.azure.com"   # e.g. https://my-openai.openai.azure.com
AZURE_OPENAI_API_KEY = "28Jo9eIRdmoCCQTzUB0upgSIZpXksrfZnmhp2Vo46UQI7QrKqE18JQQJ99BGACYeBjFXJ3w3AAABACOGMV0T"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = "text-embedding-ada-002"   # name of your embedding deployment in Azure
AZURE_OPENAI_CHAT_DEPLOYMENT = "gpt-35-turbo"                # name of your chat/completion deployment

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2024-02-15-preview"   # check portal for supported versions
)

# ---------- Azure AI Search Config ----------
SEARCH_ENDPOINT = "https://testadosearch.search.windows.net"      # e.g. https://my-search.search.windows.net
SEARCH_KEY = "23kHJYzGON0i1pw9hb7Sx9R0hBzznwgCPkFwdZspPjAzSeDIkI11"
SEARCH_INDEX = "rag-1756292528423"                        # your index name

search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=SEARCH_INDEX,
    credential=AzureKeyCredential(SEARCH_KEY)
)

# ---------- Helper function ----------
def ask_gpt(query: str):
    # Step 1: Get embedding for query
    embedding = client.embeddings.create(
        model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        input=query
    )
    vector = embedding.data[0].embedding
    print(f"Got embedding of length {len(vector)}")

    # Step 2: Query Azure AI Search with vector
    vector_query = VectorizedQuery(
        vector=vector,
        k_nearest_neighbors=3,
        fields="text_vector",
        kind="vector"  # Add the 'kind' parameter explicitly
    )

    results = search_client.search(
        search_text="",
        vector_queries=[vector_query],
        select=["chunk", "title"]
    )

    context = "\n".join([doc["chunk"] for doc in results])

    # Step 3: Ask GPT with retrieved context
    completion = client.chat.completions.create(
        model=AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Use context to answer."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        max_tokens=500
    )

    return completion.choices[0].message.content

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(query: Query):
    answer = ask_gpt(query.question)
    return {"answer": answer}

@app.get("/ask")
def ask_get(question: str):
    answer = ask_gpt(question)
    return {"answer": answer}
    
# from browser, POST ll not work, always GET, use like below with curl for POST call
#  curl -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: application/json" -d '{"question":"How do I stop AKS cluster?"}'

# for GET to work from brower, use like this - http://127.0.0.1:8000/ask?question=How+do+I+stop+AKS+cluster

# ---------- Example code for command line query and answer ----------
#if __name__ == "__main__":
#    answer = ask_gpt("Share me steps only to stop AKS cluster")
#    print(answer)
