# HR Policy Assistant (RAG)

An AI-powered Retrieval-Augmented Generation (RAG) web application designed to help employees query their company's HR policies. 

### Tech Stack
* **Frontend:** Streamlit
* **PDF Processing:** PyMuPDF
* **Embeddings:** HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`)
* **Vector Database:** FAISS
* **LLM:** Groq API 

### How to use
1. Upload a PDF of your HR Policy.
2. Type in a question (e.g., "What is the maternity leave policy?").
3. The AI retrieves the exact context and generates a precise answer based *only* on the document.