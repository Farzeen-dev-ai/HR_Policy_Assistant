import streamlit as st
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Streamlit Page Config
st.set_page_config(page_title="HR Policy Assistant", page_icon="🏢")

st.title("🏢 HR Policy Assistant")
st.write("Upload an HR Policy PDF and ask questions about your company's policies.")

# Ensure API Key is provided via Streamlit Secrets
if "GROQ_API_KEY" not in st.secrets:
    st.error("GROQ_API_KEY is missing. Please add it to Streamlit Secrets.")
    st.stop()

# Initialize Groq LLM
llm = ChatGroq(
    api_key=st.secrets["GROQ_API_KEY"],
    model="openai/gpt-oss-20b",
    temperature=0.3
)

# Extract text from PDF using PyMuPDF
def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Process document and create Vector Store
@st.cache_resource
def get_vector_store(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

# Sidebar for PDF Upload
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload HR Policy (PDF)", type=["pdf"])

# Main Application Logic
if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        # Extract and index the text
        raw_text = extract_text(uploaded_file)
        vector_store = get_vector_store(raw_text)
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        
        # Setup LangChain RAG
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an HR assistant. Answer the user's questions based ONLY on the provided HR policy context. If the answer is not in the context, say 'I cannot find this information in the uploaded policy.'\n\nContext:\n{context}"),
            ("human", "{input}")
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        st.success("PDF processed successfully! Ask your questions below.")

    # Chat Interface
    user_question = st.text_input("Ask a question about the HR Policy:")
    
    if user_question:
        with st.spinner("Searching for answers..."):
            response = rag_chain.invoke({"input": user_question})
            
            st.markdown("### Answer:")
            st.write(response["answer"])
            
            with st.expander("View Source Chunks"):
                for i, doc in enumerate(response["context"]):
                    st.write(f"**Chunk {i+1}:** {doc.page_content}")
else:
    st.info("Please upload a PDF document in the sidebar to get started.")