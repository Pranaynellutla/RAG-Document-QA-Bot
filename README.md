# 🤖 Basic Document Q&A Bot (RAG Pipeline)

Project Overview
A functional Retrieval-Augmented Generation (RAG) pipeline that allows users to ask natural language questions against a specific collection of documents and receive accurate, grounded answers with source citations.


## 🛠️ Tech Stack

Python: 3.11+

Framework: LangChain

Vector DB: FAISS

LLM: Google Gemini

UI: Streamlit

## 🏗️ Architecture Overview

Ingestion: Loads 5 PDFs from the /data directory.

Chunking: Recursive character splitting with 200-character overlap.

Embedding: Batched processing via Google Generative AI.

Retrieval: Similarity search for relevant context chunks.

Generation: LLM synthesis with filename and page number citations.

## 💡 Technical Decisions
FAISS: Chosen for local persistence and high-speed retrieval without cloud overhead.

Overlap: A 200-character overlap was implemented to prevent loss of context at chunk boundaries.

## 🚀 Setup & Execution
Install Requirements:

Bash
pip install -r requirements.txt


API Configuration:
Add your GOOGLE_API_KEY to a .env file.


Run Pipeline:

Bash
python ingest.py

streamlit run app.py

## ❓ Example Queries
"What is the impact of Bitcoin on financial systems?"

"How does science influence law?"

"Summarize the Teesside University course details."

"What are the AI internship requirements?"

"Is there information about Mars in these documents?" (Testing out-of-context handling)

## ⚠️ Known Limitations
Requires a local .env file for API keys.

Supports only PDF ingestion in the current version.

Requires re-indexing if files in the /data folder are modified.
