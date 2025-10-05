<<<<<<< HEAD
import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chat_models import ChatOpenAI

from tools.flight_finder import flight_finder
from tools.hotel_finder import hotel_finder
from tools.weather_tool import weather_tool

load_dotenv()

llm = ChatOpenAI(temperature=0, model="gpt-4")

tools = [
    Tool(name="FlightFinder", func=flight_finder, description="Finds flights to a city"),
    Tool(name="HotelFinder", func=hotel_finder, description="Finds hotels in a city"),
    Tool(name="WeatherTool", func=weather_tool, description="Gives weather updates for a city"),
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# Example use
response = agent.run("I'm planning a trip to Goa. Can you find flights, hotels, and tell me the weather?")
print(response)
=======
import streamlit as st
import os
import tempfile
import shutil
from typing import List, Dict, Any

# Core LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# File processing imports
from PyPDF2 import PdfReader
from docx import Document
import docx2txt

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Page configuration
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG Document Assistant")
st.markdown("Upload documents and ask questions about their content!")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Model selection
    model_option = st.selectbox(
        "Choose LLM",
        ["Ollama (Local)", "OpenAI (API)"],
        help="Select the language model to use for generation"
    )
    
    if model_option == "OpenAI (API)":
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key"
        )
        openai_model = st.selectbox(
            "OpenAI Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
            help="Choose the OpenAI model to use"
        )
    
    embedding_option = st.selectbox(
        "Embedding Model",
        ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-MiniLM-L3-v2"],
        help="Choose the embedding model for document encoding"
    )
    
    chunk_size = st.slider("Chunk Size", 100, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, 50)

def extract_text_from_pdf(file) -> str:
    """Extract text from PDF file"""
    try:
        pdf = PdfReader(file)
        text = ''
        for page in pdf.pages:
            text += page.extract_text() or ''
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def extract_text_from_docx(file) -> str:
    """Extract text from DOCX file"""
    try:
        text = docx2txt.process(file)
        return text.strip()
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return None

def extract_text_from_txt(file) -> str:
    """Extract text from TXT file"""
    try:
        text = file.read().decode('utf-8')
        return text.strip()
    except Exception as e:
        st.error(f"Error reading TXT: {str(e)}")
        return None

def extract_text(file) -> str:
    """Extract text from uploaded file based on extension"""
    file_extension = file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        return extract_text_from_pdf(file)
    elif file_extension == 'docx':
        return extract_text_from_docx(file)
    elif file_extension == 'txt':
        return extract_text_from_txt(file)
    else:
        st.error(f"Unsupported file type: {file_extension}")
        return None

def process_documents(files: List) -> List[Dict[str, Any]]:
    """Process uploaded files and return documents with metadata"""
    documents = []
    
    for file in files:
        with st.spinner(f"Processing {file.name}..."):
            text = extract_text(file)
            
            if text:
                # Create document with metadata
                doc = {
                    'content': text,
                    'metadata': {
                        'source': file.name,
                        'type': file.name.split('.')[-1].lower()
                    }
                }
                documents.append(doc)
                st.success(f"✅ Processed {file.name}")
    
    return documents

def create_vectorstore(documents: List[Dict[str, Any]], embedding_model: str):
    """Create vector store from documents"""
    try:
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=f"sentence-transformers/{embedding_model}",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        
        # Prepare texts and metadata for LangChain
        texts = [doc['content'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        # Split documents into chunks
        chunks = text_splitter.create_documents(texts, metadatas=metadatas)
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        
        return vectorstore
        
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def initialize_llm(model_option: str):
    """Initialize the language model"""
    if model_option == "Ollama (Local)":
        try:
            return Ollama(model="mistral")  # or "llama2" based on your setup
        except Exception as e:
            st.error(f"Error initializing Ollama: {str(e)}")
            return None
    else:
        # OpenAI would be implemented here if API key is provided
        st.warning("OpenAI integration not fully implemented in this example")
        return None

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Document Upload")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload your documents (PDF, DOCX, TXT)",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'txt']
    )
    
    if uploaded_files:
        if st.button("Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                # Process documents
                documents = process_documents(uploaded_files)
                
                if documents:
                    # Create vector store
                    vectorstore = create_vectorstore(documents, embedding_option)
                    
                    if vectorstore:
                        st.session_state.vectorstore = vectorstore
                        st.session_state.documents_processed = True
                        st.success(f"✅ Processed {len(documents)} documents into {len(vectorstore.get()['documents'])} chunks!")
                        
                        # Show document info
                        with st.expander("Document Information"):
                            for doc in documents:
                                st.write(f"📄 {doc['metadata']['source']} ({doc['metadata']['type'].upper()})")
                                st.write(f"📏 {len(doc['content'])} characters")

with col2:
    st.header("Chat")
    
    # Chat interface (only show if documents are processed)
    if st.session_state.documents_processed and st.session_state.vectorstore:
        user_question = st.text_input("Ask a question about your documents:")
        
        if user_question:
            if st.button("Get Answer"):
                with st.spinner("Searching and generating answer..."):
                    try:
                        # Initialize LLM
                        llm = initialize_llm(model_option)
                        
                        if llm:
                            # Create retrieval chain
                            qa_chain = RetrievalQA.from_chain_type(
                                llm=llm,
                                chain_type="stuff",
                                retriever=st.session_state.vectorstore.as_retriever(
                                    search_kwargs={"k": 3}
                                ),
                                return_source_documents=True
                            )
                            
                            # Get answer
                            result = qa_chain({"query": user_question})
                            
                            # Display answer
                            st.subheader("Answer:")
                            st.write(result["result"])
                            
                            # Show source documents
                            with st.expander("Source Documents"):
                                for i, doc in enumerate(result["source_documents"]):
                                    st.write(f"**Source {i+1}:** {doc.metadata.get('source', 'Unknown')}")
                                    st.write(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
                                    st.write("---")
                            
                            # Add to chat history
                            st.session_state.chat_history.append({
                                "question": user_question,
                                "answer": result["result"],
                                "sources": len(result["source_documents"])
                            })
                        
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
    
    # Show chat history
    if st.session_state.chat_history:
        with st.expander("Chat History"):
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
                st.write(f"**Q:** {chat['question']}")
                st.write(f"**A:** {chat['answer'][:100]}..." if len(chat['answer']) > 100 else f"**A:** {chat['answer']}")
                st.write(f"📚 Sources: {chat['sources']}")
                st.write("---")

# Footer
st.markdown("---")
st.markdown("*Built with LangChain, ChromaDB, and Streamlit*")
    
>>>>>>> 9389cb1 (Initial commit)
