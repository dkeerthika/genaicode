import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# Azure OpenAI credentials
GPT4O_MINI_ENDPOINT = "https://bfsi-genai-demo.openai.azure.com/"
GPT4O_MINI_MODEL_NAME = "gpt-4o-mini"
GPT4O_MINI_DEPLOYMENT = "bfsi-genai-demo-gpt-4o-mini"
GPT4O_MINI_API_KEY = "406a7e3789194dcab310ead3ee2fb035"
GPT4O_MINI_API_VERSION = "2024-12-01-preview"

# -- Streamlit UI Setup
st.set_page_config(page_title="📘 GPT-4o Mini RAG App", layout="centered")
st.title("📘 Textbook Q&A Using GPT-4o Mini (Azure OpenAI)")

# -- Load Pre-built Vector Store
@st.cache_resource
def load_vector_store(persist_path="./chroma_db"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=persist_path, embedding_function=embeddings)
    return vectordb

# -- AI Model
llm = AzureChatOpenAI(
    api_version=GPT4O_MINI_API_VERSION,
    azure_endpoint=GPT4O_MINI_ENDPOINT,
    api_key=GPT4O_MINI_API_KEY,
    deployment_name=GPT4O_MINI_DEPLOYMENT
)

# -- RAG Prompt Template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful school teacher. Use the following CONTEXT from a science textbook to answer the QUESTION as clearly and factually as possible.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTION:
- Only use the context provided above to answer.
- If the answer is not found in context, respond with: "I don't know the answer."

Answer:
"""
)

# -- Create Chain
chain = prompt | llm  # Prompt passed to LLM directly

# -- Text Input
query = st.text_input("🔎 Ask a textbook-related question:")

if query:
    vectordb = load_vector_store()
    retriever = vectordb.as_retriever()
    
    with st.spinner("Searching textbook..."):
        results = retriever.get_relevant_documents(query)  # 🔧 FIXED this line
        context = "\n".join([doc.page_content for doc in results])
        response = chain.invoke({"context": context, "question": query})

    st.subheader("📘 Answer:")
    st.write(response.content)
