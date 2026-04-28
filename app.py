import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

# NOTICE: We have removed 'langchain.chains' entirely.
# We are using 'langchain_classic' which is the 2026 standard.
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
# Load Environment Variables
load_dotenv()

# App Configuration
st.set_page_config(page_title="AI Intern RAG Bot", layout="wide")
st.title("📄 Multi-Document Q&A Bot")
st.markdown("This bot uses a RAG pipeline to answer questions based on your local documents.")

# Component 1: Load the Persistent Vector Store
@st.cache_resource
def get_retriever():
    # CRITICAL: This must match the model you used in ingest.py!
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # allow_dangerous_deserialization is required for loading local FAISS files
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return vector_store.as_retriever(search_kwargs={"k": 3})

# Check if index exists
if not os.path.exists("faiss_index"):
    st.warning("⚠️ Vector store not found. Please run 'python ingest.py' in your terminal first.")
else:
    # Component 2: Initialize LLM & Chain
    # We use gemini-1.5-flash for speed and efficiency
    # Force the use of the stable v1 API to avoid the v1beta 404 error
    # Simplified initialization to avoid Pydantic validation errors
    # Change this line in your app.py:
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.2)
    
    system_prompt = (
        "You are a professional assistant. Answer the user's question using ONLY the context provided below. "
        "If the answer is not in the context, say: 'I am sorry, but the provided documents do not contain information regarding this.' "
        "Always cite the source filename and page number in your response."
        "\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Create the RAG Pipeline
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retriever = get_retriever()
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # UI Interaction
    user_query = st.text_input("Ask a question about the documents:")

    if user_query:
        with st.spinner("Analyzing documents..."):
            # Invoke the chain
            response = rag_chain.invoke({"input": user_query})
            
            # Show Answer
            st.markdown("### 🤖 Answer")
            st.write(response["answer"])

            # Requirement 4.7: Show source chunks used for citations
            st.markdown("---")
            with st.expander("🔍 View Source Material & Citations"):
                for i, doc in enumerate(response["context"]):
                    source_name = doc.metadata.get('source', 'Unknown')
                    # PDF pages in LangChain are 0-indexed, so we add 1 for the user
                    page_num = doc.metadata.get('page', 0) + 1 
                    st.markdown(f"**Chunk {i+1} from: `{source_name}` (Page {page_num})**")
                    st.info(doc.page_content)