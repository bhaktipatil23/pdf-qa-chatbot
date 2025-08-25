import os
import tempfile
import asyncio
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA


try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
    
# Make sure Streamlit writes configs inside the repo, not root
os.environ["STREAMLIT_HOME"] = os.path.join(os.getcwd(), ".streamlit")

# ---------------- Streamlit Page Setup ----------------
st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")
st.title("PDF Analysis chatbot using RAG")

# ---------------- API Key Setup ----------------
# The app will not run if we don't have a Gemini API Key in environment variables.
apiKey = os.getenv("GEMINI_API_KEY")
if not apiKey:
    st.error("Please set GEMINI_API_KEY as an environment variable or in Streamlit secrets.")
    st.stop()

# ---------------- Sidebar for File Upload ----------------
st.sidebar.header("Upload PDFs")
uploaded_files = st.sidebar.file_uploader(
    "Select PDF files to upload", type=["pdf"], accept_multiple_files=True
)

# Show uploaded file names in sidebar
if uploaded_files:
    st.sidebar.subheader("Uploaded Files:")
    for f in uploaded_files:
        st.sidebar.write(f"- {f.name}")

# ---------------- Build Index ----------------
# When user clicks the button, process PDFs into chunks and build vector index
if st.sidebar.button("Build Index"):
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one PDF before building index.")
        st.stop()

    docs = []
    for uploaded_file in uploaded_files:
        # Save the uploaded file temporarily so PyPDFLoader can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Load PDF and extract text
        loader = PyPDFLoader(tmp_path)
        docs.extend(loader.load())

    # Break documents into smaller chunks for better retrieval
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    # Convert chunks into embeddings and store them in FAISS (vector database)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=apiKey)
    vectordb = FAISS.from_documents(chunks, embeddings)

    # Store retriever in session so it can be reused in chat
    st.session_state.retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    st.sidebar.success(f"Index created with {len(chunks)} chunks from {len(uploaded_files)} file(s).")

# ---------------- Chat Section ----------------
# This section creates a chatbot-like interface using chat history
if "retriever" in st.session_state:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=apiKey, temperature=0.2)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=st.session_state.retriever,
        chain_type="stuff",
        return_source_documents=True,
    )

    # Store chat messages so they persist between user inputs
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show old messages from history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input box for user to ask questions
    if query := st.chat_input("Ask something about your PDFs..."):
        # Save user query into chat history
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Run the question against the retrieval+LLM chain
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = qa_chain.invoke(query)
                response = result["result"]
                st.markdown(response)

        # Save assistant reply into chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Optionally show which document chunks were used to answer
        with st.expander("Sources"):
            for i, doc in enumerate(result["source_documents"], 1):
                st.markdown(f"Source {i} (Page {doc.metadata.get('page', 'N/A')})")
                st.write(doc.page_content[:500] + "...")
