import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
import time
import tempfile

# Load environment variables
load_dotenv()
pinecone_key = os.getenv("PINECONE_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

# Initialize models
@st.cache_resource
def get_models():
    chat = ChatGroq(api_key=groq_key, model_name="llama3-70b-8192")
    embed_model = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_key)
    return chat, embed_model

# Load PDF from uploaded file
def read_uploaded_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name
    
    try:
        loader = PyPDFLoader(temp_path)
        documents = loader.load()
        st.success(f"Successfully loaded {len(documents)} pages from the PDF")
        return documents
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return []
    finally:
        # Clean up the temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Chunking
def chunk_data(docs, chunk_size=800, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(docs)
    return chunks

# Pinecone setup function
def setup_pinecone(index_name, namespace, embed_model, documents):
    pc = Pinecone(api_key=pinecone_key)
    cloud = os.getenv('PINECONE_CLOUD', 'aws')
    region = os.getenv('PINECONE_REGION', 'us-east-1')
    spec = ServerlessSpec(cloud=cloud, region=region)
    
    # Check if index exists, if so delete it
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)
        st.info(f"Deleted existing index: {index_name}")
    
    # Create new index
    pc.create_index(index_name, dimension=3072, metric='cosine', spec=spec)
    st.info(f"Created new index: {index_name}")
    
    # Give Pinecone a moment to initialize
    time.sleep(1)
    
    # Create vector store from documents
    docsearch = PineconeVectorStore.from_documents(
        documents=documents,
        index_name=index_name,
        embedding=embed_model,
        namespace=namespace
    )
    
    # Allow time for indexing
    time.sleep(1)
    
    # Create vectorstore retriever
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embed_model,
        namespace=namespace,
    )
    retriever = vectorstore.as_retriever()
    
    return retriever

# Create QA chain
def create_qa_chain(chat, retriever):
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# Streamlit UI
st.title("PDF Question Answering System")
st.write("Upload a PDF document and ask questions about its content")

# File upload section
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    chunk_size = st.slider("Chunk Size", min_value=500, max_value=5000, value=3000, step=100)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=50, step=10)
    index_name = st.text_input("Pinecone Index Name", value="eguins")
    namespace = st.text_input("Pinecone Namespace", value="espacio")

# Process the uploaded file
if uploaded_file is not None:
    # Get models
    chat, embed_model = get_models()
    
    # Display sample embedding dimension
    with st.expander("View Embedding Dimension"):
        embedding_sample = embed_model.embed_query('hello')
        st.write(f"Embedding dimension: {len(embedding_sample)}")
    
    # Process button
    if st.button("Process Document"):
        with st.spinner("Loading and processing document..."):
            # Load document
            documents = read_uploaded_pdf(uploaded_file)
            
            if documents:
                # Chunk data
                with st.spinner("Chunking document..."):
                    chunks = chunk_data(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    st.success(f"Created {len(chunks)} chunks from the document")
                
                # Setup Pinecone and create vector store
                with st.spinner("Setting up vector database..."):
                    retriever = setup_pinecone(index_name, namespace, embed_model, chunks)
                    st.success("Vector database setup complete")
                
                # Create QA chain
                with st.spinner("Initializing QA system..."):
                    qa_chain = create_qa_chain(chat, retriever)
                    st.session_state.qa_chain = qa_chain
                    st.success("QA system ready!")
    
    # QA interface
    if 'qa_chain' in st.session_state:
        st.header("Ask Questions")
        query = st.text_input("Enter your question:")
        
        if st.button("Get Answer"):
            if query:
                with st.spinner("Searching for answer..."):
                    result = st.session_state.qa_chain({"query": query})
                    
                    # Display answer
                    st.header("Answer")
                    st.write(result['result'])
                    
                    # Display sources
                    with st.expander("View Source Documents"):
                        for i, doc in enumerate(result['source_documents']):
                            st.subheader(f"Source {i+1}")
                            st.write(doc.page_content)
                            st.write("---")
else:
    st.info("Please upload a PDF file to get started")

# Sample questions section
if 'qa_chain' in st.session_state:
    st.header("Sample Questions")
    sample_questions = [
        "Where does the person live?",
        "What education does the person have?",
        "List all companies where the person has worked",
        "What is the person's phone number?",
        "What skills does the person have?"
    ]
    
    selected_question = st.selectbox("Try a sample question:", [""] + sample_questions)
    
    if selected_question and st.button("Ask Sample Question"):
        with st.spinner("Searching for answer..."):
            result = st.session_state.qa_chain({"query": selected_question})
            
            # Display answer
            st.header("Answer")
            st.write(result['result'])
            
            # Display sources
            with st.expander("View Source Documents"):
                for i, doc in enumerate(result['source_documents']):
                    st.subheader(f"Source {i+1}")
                    st.write(doc.page_content)
                    st.write("---")