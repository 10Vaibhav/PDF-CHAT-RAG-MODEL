import streamlit as st
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
import os

st.set_page_config(
    page_title="PDF RAG Chat",
    page_icon="📃",
    layout="wide"
)
if "api_key_configured" not in st.session_state:
    st.session_state.api_key_configured = False
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""

# API Key Configuration Page
def show_api_key_page():
    st.title("🔑 API Key Configuration")
    st.markdown("Please enter your OpenAI API key to continue")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.info("💡 Your API key will only be stored in this session and won't be saved permanently.")
        
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Enter your OpenAI API key"
        )
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("✅ Configure", type="primary", use_container_width=True):
                if api_key and api_key.startswith("sk-"):
                    # Set environment variable
                    os.environ["OPENAI_API_KEY"] = api_key
                    st.session_state.openai_api_key = api_key
                    st.session_state.api_key_configured = True
                    st.success("✅ API Key configured successfully!")
                    st.rerun()
                else:
                    st.error("❌ Please enter a valid OpenAI API key (starts with 'sk-')")
        
        with col_b:
            # Check if .env file exists
            if Path(".env").exists():
                if st.button("📁 Load from .env", use_container_width=True):
                    from dotenv import load_dotenv
                    load_dotenv()
                    env_key = os.getenv("OPENAI_API_KEY")
                    if env_key:
                        st.session_state.openai_api_key = env_key
                        st.session_state.api_key_configured = True
                        st.success("✅ API Key loaded from .env file!")
                        st.rerun()
                    else:
                        st.error("❌ No OPENAI_API_KEY found in .env file")
        
        st.divider()
        
        with st.expander("ℹ️ How to get an OpenAI API key?"):
            st.markdown("""
            1. Go to [OpenAI Platform](https://platform.openai.com/)
            2. Sign up or log in to your account
            3. Navigate to API keys section
            4. Create a new API key
            5. Copy and paste it here
            """)

@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=st.session_state.openai_api_key)

@st.cache_resource
def get_embedding_model():
    return OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=st.session_state.openai_api_key
    )

def index_pdf(pdf_file):
    """Index the uploaded PDF file into Qdrant"""
    try:
        temp_path = Path("temp_upload.pdf")
        with open(temp_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        
        loader = PyPDFLoader(file_path=str(temp_path))
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=400,
        )
        chunks = text_splitter.split_documents(documents=docs)

        embedding_model = get_embedding_model()
        vector_store = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embedding_model,
            url="http://localhost:6333",
            collection_name="pdf_rag"
        )
        
        # Clean up temp file
        temp_path.unlink()
        
        return True, len(chunks)
    except Exception as e:
        return False, str(e)


@st.cache_resource
def get_vector_store():
    try:
        embedding_model = get_embedding_model()
        return QdrantVectorStore.from_existing_collection(
            url="http://localhost:6333",
            collection_name="pdf_rag",
            embedding=embedding_model
        )
    except Exception as e:
        return None

def query_rag(user_query, vector_db):
    """Query the RAG system"""
    try:
        search_results = vector_db.similarity_search(query=user_query, k=4)

        context = "\n\n\n".join([
            f"Page Content: {result.page_content}\n"
            f"Page Number: {result.metadata.get('page_label', 'N/A')}\n"
            f"File Location: {result.metadata.get('source', 'N/A')}"
            for result in search_results
        ])

        system_prompt = f"""
You are a helpful AI Assistant who answers user queries based on the available context retrieved from a PDF file along with page contents and page number.

You should only answer the user based on the following context and navigate the user to open the right page number to know more.

Context:
{context}
"""
        
        openai_client = get_openai_client()
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
        )
        
        return response.choices[0].message.content, search_results
    except Exception as e:
        return f"Error: {str(e)}", []

# Main app
def main_app():
    st.title("📃 PDF RAG Chat Assistant")
    st.markdown("Ask questions about your PDF documents!")
    
    with st.sidebar:
        st.header("📄 Document Management")
        
        st.success("✅ API Key Configured")
        if st.button("🔄 Change API Key"):
            st.session_state.api_key_configured = False
            st.session_state.openai_api_key = ""
            st.cache_resource.clear()
            st.rerun()
        
        st.divider()
        
        uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
        
        if uploaded_file:
            if st.button("Index PDF", type="primary"):
                with st.spinner("Indexing PDF..."):
                    success, result = index_pdf(uploaded_file)
                    if success:
                        st.success(f"✅ PDF indexed successfully! ({result} chunks created)")
                        st.cache_resource.clear()
                    else:
                        st.error(f"❌ Error indexing PDF: {result}")
        
        st.divider()
        st.markdown("### 📊 System Info")
        st.info("Make sure Qdrant is running on localhost:6333")
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📖 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:**")
                        st.text(source.page_content[:200] + "...")
                        st.caption(f"Page: {source.metadata.get('page_label', 'N/A')}")
                        st.divider()

    if prompt := st.chat_input("Ask a question about your PDF..."):
        vector_db = get_vector_store()
        
        if vector_db is None:
            st.error("❌ No indexed documents found. Please upload and index a PDF first.")
            return
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, sources = query_rag(prompt, vector_db)
                st.markdown(response)
                
                if sources:
                    with st.expander("📖 View Sources"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**Source {i}:**")
                            st.text(source.page_content[:200] + "...")
                            st.caption(f"Page: {source.metadata.get('page_label', 'N/A')}")
                            st.divider()
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources
        })

def main():
    if not st.session_state.api_key_configured:
        show_api_key_page()
    else:
        main_app()

if __name__ == "__main__":
    main()