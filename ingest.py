import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def run_ingestion():
    # 1. Load Documents
    print("--- Starting Ingestion ---")
    if not os.path.exists("./data"):
        print("Error: 'data' folder not found. Create it and add PDFs.")
        return

    loader = DirectoryLoader('./data', glob="./*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from PDFs.")

    # 2. Chunking (Requirement 4.2)
    # We use Recursive splitter to keep paragraphs together
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    # 3. Embedding & Vector Store (Requirement 4.3 & 4.4)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    print("Generating embeddings and saving to disk...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Save the database locally so we don't have to re-index
    vector_store.save_local("faiss_index")
    print("--- Ingestion Complete! 'faiss_index' folder created. ---")

if __name__ == "__main__":
    run_ingestion()