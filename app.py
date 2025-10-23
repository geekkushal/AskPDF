import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# ✅ Updated LangChain v0.2+ imports
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_mistralai import ChatMistralAI

# Custom UI templates
from htmlTemplates import css, bot_template, user_template


# ========== PDF Processing Functions ==========
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)


# ========== Vector Store ==========
def get_vectorstore(text_chunks):
    # Using a lightweight and efficient Hugging Face model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


# ========== Conversation Setup ==========
def get_conversation_chain(vectorstore):
    llm = ChatMistralAI(model="open-mistral-7b", temperature=0.7)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )


# ========== Chat Handler ==========
def handle_userinput(user_question):
    try:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        # Create a container for chat messages with better styling
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:  # User message
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(message.content)
                else:  # Bot message
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(message.content)
    except Exception as e:
        if "429" in str(e) or "capacity exceeded" in str(e).lower():
            st.error("🚫 **API Capacity Exceeded**: The Mistral model is currently at capacity. Please try again in a few minutes, or consider upgrading your Mistral plan.")
            st.info("💡 **Tip**: You can also try switching to a smaller model like `open-mistral-7b` in the code for better availability.")
        elif "401" in str(e) or "unauthorized" in str(e).lower():
            st.error("🔑 **Authentication Error**: Please check your Mistral API key in the .env file.")
        else:
            st.error(f"❌ **Error**: {str(e)}")
            st.info("🔄 Please try again or check your internet connection.")


# ========== Streamlit App ==========
def main():
    load_dotenv()
    
    # Check for Mistral API key
    if not os.getenv("MISTRAL_API_KEY"):
        st.error("❌ Please set your MISTRAL_API_KEY in your .env file")
        st.info("ℹ️ Get your API key from: https://console.mistral.ai/")
        st.stop()
    
    # Enhanced page configuration
    st.set_page_config(
        page_title="PDF Chat Assistant", 
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    st.write(css, unsafe_allow_html=True)
    
    # Additional modern styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e1e5e9;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin: 1rem 0;
    }
    .upload-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .feature-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Session state setup
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Main header with enhanced styling
    st.markdown("""
    <div class="main-header">
        <h1>📚 Ask PDF : By Kushal Ajwani </h1>
        <p>Upload your PDFs and start asking questions powered by AI</p>
    </div>
    """, unsafe_allow_html=True)

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat interface section
        st.markdown("### 💬 Chat Interface")
        
        # Status indicator
        if st.session_state.conversation:
            st.success("✅ Ready to chat! Documents are processed and loaded.")
        else:
            st.info("ℹ️ Upload and process your PDFs to start chatting.")
        
        # Question input with enhanced styling
        user_question = st.text_input(
            "Ask a question about your documents:",
            placeholder="e.g., What is the main topic discussed in the documents?",
            help="Type your question here and press Enter"
        )
        
        # Handle user input
        if user_question:
            if st.session_state.conversation:
                with st.container():
                    st.markdown("### 📋 Conversation")
                    handle_userinput(user_question)
            else:
                st.warning("⚠️ Please upload and process PDFs first using the sidebar.")
        
        # Show chat history if available
        elif st.session_state.chat_history:
            with st.container():
                st.markdown("### 📋 Previous Conversation")
                for i, message in enumerate(st.session_state.chat_history):
                    if i % 2 == 0:  # User message
                        with st.chat_message("user", avatar="👤"):
                            st.markdown(message.content)
                    else:  # Bot message
                        with st.chat_message("assistant", avatar="🤖"):
                            st.markdown(message.content)
    
    with col2:
        # Info panel
        st.markdown("### ℹ️ How it works")
        with st.expander("📖 Instructions", expanded=True):
            st.markdown("""
            1. **Upload PDFs** 📄 - Use the sidebar to upload one or more PDF files
            2. **Process Documents** ⚙️ - Click 'Process' to analyze your documents  
            3. **Ask Questions** ❓ - Type questions about your documents
            4. **Get Answers** 🎯 - Receive AI-powered responses based on your content
            """)
        
        with st.expander("✨ Features"):
            st.markdown("""
            - **Multi-PDF Support** - Upload multiple documents at once
            - **Intelligent Search** - AI finds relevant information across all documents
            - **Conversation Memory** - Maintains context throughout the chat
            - **Real-time Processing** - Fast document analysis and response generation
            """)

    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("## 📁 Document Management")
        st.divider()
        
        # Upload section with enhanced styling
        st.markdown("""
        <div class="upload-section">
            <h3>📤 Upload Documents</h3>
            <p>Select one or more PDF files to get started</p>
        </div>
        """, unsafe_allow_html=True)
        
        pdf_docs = st.file_uploader(
            "Choose PDF files",
            accept_multiple_files=True,
            type=['pdf'],
            help="You can upload multiple PDF files at once"
        )
        
        # Show uploaded files info
        if pdf_docs:
            st.markdown("### 📋 Uploaded Files")
            for i, pdf in enumerate(pdf_docs, 1):
                st.markdown(f"**{i}.** {pdf.name}")
                st.caption(f"Size: {pdf.size / 1024:.1f} KB")
            
            st.divider()
        
        # Process button with enhanced styling
        if st.button("🚀 Process Documents", type="primary", use_container_width=True):
            if pdf_docs:
                with st.spinner("🔄 Processing your documents..."):
                    # Show processing steps
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Extract text
                    status_text.text("📖 Extracting text from PDFs...")
                    progress_bar.progress(25)
                    raw_text = get_pdf_text(pdf_docs)
                    
                    # Split into chunks
                    status_text.text("✂️ Splitting text into chunks...")
                    progress_bar.progress(50)
                    chunks = get_text_chunks(raw_text)
                    
                    # Embed and store vectors
                    status_text.text("🧠 Creating knowledge base...")
                    progress_bar.progress(75)
                    vectorstore = get_vectorstore(chunks)
                    
                    # Setup conversation chain
                    status_text.text("⚡ Setting up conversation...")
                    progress_bar.progress(100)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                
                st.success("✅ Documents processed successfully!")
                st.balloons()
            else:
                st.error("❌ Please upload at least one PDF file.")
        
        st.divider()
        
        # Additional sidebar info
        with st.expander("🛠️ Technical Details"):
            st.markdown("""
            **Model:** Open Mistral 7B (Free)  
            **Embeddings:** MiniLM-L6-v2 (Optimized)   
            **Vector Store:** FAISS  
            **Chunk Size:** 1000 characters  
            **Chunk Overlap:** 200 characters  
            """)
        
        with st.expander("💡 Tips"):
            st.markdown("""
            - **Clear Questions:** Ask specific questions for better results
            - **Multiple Documents:** The AI can find information across all uploaded PDFs
            - **Follow-up Questions:** The system remembers your conversation context
            - **Document Quality:** Better quality PDFs give better results
            """)
        
        


if __name__ == '__main__':
    main()
