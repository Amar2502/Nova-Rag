import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

import tempfile
import os
from stt import transcribe_chunk
from prompt import RAG_TEMPLATE
from env import get_api


# -------------------------------------------
# Streamlit UI Configuration
# -------------------------------------------
st.set_page_config(
    page_title="NOVA RAG",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

api_key = get_api()

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background with gradient */
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
        background-attachment: fixed;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0c29 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Sidebar title */
    [data-testid="stSidebar"] h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2rem !important;
        margin-bottom: 2rem;
    }
    
    /* Text styling */
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-weight: 600;
    }
    
    p, label, div, span {
        color: #E4E4E4 !important;
    }
    
    /* Main title with gradient */
    .main h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem !important;
        font-weight: 700;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.05);
        border: 2px dashed rgba(102, 126, 234, 0.5);
        border-radius: 12px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(102, 126, 234, 0.8);
        background: rgba(255, 255, 255, 0.08);
    }
    
    /* Buttons with gradient */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white !important;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Chat input styling */
    .stChatInput>div>div {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    
    .stChatInput>div>div:focus-within {
        border-color: rgba(102, 126, 234, 0.8);
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* User message */
    [data-testid="stChatMessage"][data-testid*="user"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    /* Assistant message */
    [data-testid="stChatMessage"][data-testid*="assistant"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Info box */
    .stAlert {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 10px;
        color: #E4E4E4 !important;
    }
    
    /* Success message */
    .stSuccess {
        background: rgba(76, 175, 80, 0.1);
        border: 1px solid rgba(76, 175, 80, 0.3);
        border-radius: 10px;
    }
    
    /* Status container */
    [data-testid="stStatus"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    /* Markdown in chat - properly formatted */
    [data-testid="stChatMessage"] .stMarkdown {
        color: #E4E4E4 !important;
    }
    
    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3 {
        color: #FFFFFF !important;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    [data-testid="stChatMessage"] code {
        background: rgba(0, 0, 0, 0.3);
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        color: #ff79c6 !important;
        font-family: 'Courier New', monospace;
    }
    
    [data-testid="stChatMessage"] pre {
        background: rgba(0, 0, 0, 0.4);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid rgba(102, 126, 234, 0.3);
        overflow-x: auto;
    }
    
    [data-testid="stChatMessage"] ul,
    [data-testid="stChatMessage"] ol {
        margin-left: 1.5rem;
        color: #E4E4E4 !important;
    }
    
    [data-testid="stChatMessage"] li {
        margin-bottom: 0.5rem;
        color: #E4E4E4 !important;
    }
    
    [data-testid="stChatMessage"] blockquote {
        border-left: 3px solid rgba(102, 126, 234, 0.8);
        padding-left: 1rem;
        margin: 1rem 0;
        color: #B0B0B0 !important;
        font-style: italic;
    }
    
    [data-testid="stChatMessage"] a {
        color: #667eea !important;
        text-decoration: none;
    }
    
    [data-testid="stChatMessage"] a:hover {
        text-decoration: underline;
    }
    
    /* Stats boxes in sidebar */
    .stat-box {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #B0B0B0;
        margin-top: 0.5rem;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.3);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Avatar icons */
    [data-testid="stChatMessageAvatarUser"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    [data-testid="stChatMessageAvatarAssistant"] {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%) !important;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------
# Session State Initialization
# -------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "mode" not in st.session_state:
    st.session_state.mode = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = "Ollama (gemma3)"


# -------------------------------------------
# PDF Processing
# -------------------------------------------
def process_pdf(uploaded_file):
    print("[DEBUG] Starting PDF processing")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        pdf_path = tmp.name
    print(f"[DEBUG] PDF saved to temp path: {pdf_path}")

    with st.status("📄 Processing PDF...", expanded=True) as status:
        st.write("Loading document...")
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        print(f"[DEBUG] Loaded {len(docs)} PDF documents")

        st.write("Splitting into chunks...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        print(f"[DEBUG] Created {len(chunks)} PDF chunks")

        st.write(f"Creating embeddings for {len(chunks)} chunks...")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        st.write("Building vector store...")
        vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
        print("[DEBUG] PDF vector store built")
        
        status.update(label="✅ PDF processed successfully!", state="complete", expanded=False)

    os.unlink(pdf_path)
    print(f"[DEBUG] Removed temp PDF: {pdf_path}")
    return vectorstore


# -------------------------------------------
# Audio Processing
# -------------------------------------------
def process_audio(uploaded_file):
    print("[DEBUG] Starting audio processing")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_file.getvalue())
        audio_path = tmp.name
    print(f"[DEBUG] Audio saved to temp path: {audio_path}")

    with st.status("🎵 Transcribing Audio...", expanded=True) as status:
        st.write("Converting speech to text...")
        text_path = transcribe_chunk(audio_path)
        print(f"[DEBUG] Transcription saved to: {text_path}")

        st.write("Loading transcription...")
        loader = TextLoader(text_path)
        docs = loader.load()
        print(f"[DEBUG] Loaded {len(docs)} transcription documents")
        
        st.write("Splitting into chunks...")
        splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        print(f"[DEBUG] Created {len(chunks)} transcription chunks")

        st.write(f"Creating embeddings for {len(chunks)} chunks...")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        st.write("Building vector store...")
        vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
        print("[DEBUG] Audio vector store built")
        
        status.update(label="✅ Audio transcribed successfully!", state="complete", expanded=False)

    os.unlink(audio_path)
    print(f"[DEBUG] Removed temp audio: {audio_path}")
    return vectorstore


# -------------------------------------------
# LLM Factory
# -------------------------------------------
def get_llm(provider_choice):
    """
    Return the appropriate chat model based on user selection.
    """
    print(f"[DEBUG] LLM provider selected: {provider_choice}")
    if provider_choice == "Gemini 2.5 Flash":
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            google_api_key=api_key,
        )
    # Default to Ollama
    return ChatOllama(model="gemma3")


# -------------------------------------------
# RAG Response
# -------------------------------------------
def get_response(question, vectorstore):
    with st.status("🤔 Thinking...", expanded=False) as status:
        print(f"[DEBUG] Received question: {question}")
        st.write("Searching relevant documents...")
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8}
        )
        docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs])
        print(f"[DEBUG] Context: {context}")
        print(f"[DEBUG] Retrieved {len(docs)} documents for context")

        st.write("Generating response...")
        llm = get_llm(st.session_state.llm_provider)
        print("[DEBUG] LLM initialized, starting streaming response")

        prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
        final_prompt = prompt.format(context=context, question=question)

        # --- STREAMING UI ---
        response_box = st.empty()     # placeholder for streaming
        answer = ""
        chunk_count = 0

        for chunk in llm.stream(final_prompt):
            if chunk.content:
                answer += chunk.content
                response_box.markdown(answer)   # update UI live
                chunk_count += 1
        print(f"[DEBUG] Completed streaming with {chunk_count} chunks")
        print(f"[DEBUG] Answer: {answer}")

        status.update(label="✅ Response generated!", state="complete", expanded=False)

    return answer

# -------------------------------------------
# Sidebar (Uploads)
# -------------------------------------------
with st.sidebar:
    st.markdown("# ✨ NOVA RAG")
    st.markdown("---")
    
    st.markdown("### 🧠 Model")
    st.session_state.llm_provider = st.selectbox(
        "Choose a provider",
        options=["Ollama (gemma3)", "Gemini 2.5 Flash"],
        index=0 if st.session_state.llm_provider == "Ollama (gemma3)" else 1,
        help="Select which model to use for responses",
    )
    st.markdown("---")
    
    # Stats section
    if st.session_state.vectorstore is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{len(st.session_state.messages)}</div>
                <div class="stat-label">Messages</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            mode_icon = "📄" if st.session_state.mode == "pdf" else "🎵"
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{mode_icon}</div>
                <div class="stat-label">{st.session_state.mode.upper()}</div>
            </div>
            """, unsafe_allow_html=True)
        
        if st.session_state.document_name:
            st.info(f"📁 *Current:* {st.session_state.document_name}")
        
        st.markdown("---")

    st.markdown("### 📤 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF or Audio file",
        type=["pdf", "mp3", "wav", "mpeg"],
        help="Upload a PDF document or audio file to start chatting"
    )

    if uploaded_file:
        filetype = uploaded_file.type

        if filetype == "application/pdf":
            if st.button("🚀 Process PDF", use_container_width=True):
                with st.spinner("Processing..."):
                    st.session_state.vectorstore = process_pdf(uploaded_file)
                    st.session_state.mode = "pdf"
                    st.session_state.document_name = uploaded_file.name
                    st.session_state.messages = []  # Clear previous messages
                    st.success(f"✅ Processed {uploaded_file.name}")
                    st.rerun()

        elif filetype in ["audio/mpeg", "audio/mp3", "audio/wav"]:
            if st.button("🚀 Process Audio", use_container_width=True):
                with st.spinner("Processing..."):
                    st.session_state.vectorstore = process_audio(uploaded_file)
                    st.session_state.mode = "audio"
                    st.session_state.document_name = uploaded_file.name
                    st.session_state.messages = []  # Clear previous messages
                    st.success(f"✅ Processed {uploaded_file.name}")
                    st.rerun()
    
    st.markdown("---")
    
    # Clear chat button
    if st.session_state.vectorstore is not None:
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 1rem; color: #B0B0B0; font-size: 0.8rem;'>
        <p>Powered by LangChain & Ollama</p>
        <p>✨ NOVA RAG v2.0</p>
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------
# Main Chat UI
# -------------------------------------------
st.title("💬 Chat with NOVA")

if st.session_state.vectorstore is None:
    st.markdown("""
    <div style='text-align: center; padding: 3rem; margin-top: 2rem;'>
        <h2 style='color: #667eea;'>👋 Welcome to NOVA RAG</h2>
        <p style='font-size: 1.2rem; margin-top: 1rem; color: #B0B0B0;'>
            Upload a PDF or audio file from the sidebar to start an intelligent conversation
        </p>
        <p style='margin-top: 2rem; color: #888;'>
            📄 Support for PDF documents<br>
            🎵 Support for audio transcription<br>
            🤖 Powered by advanced AI models
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    user_input = st.chat_input("💭 Ask me anything about your document...")

    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        response = get_response(user_input, st.session_state.vectorstore)

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()