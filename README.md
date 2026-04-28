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
Chunking Strategy: 

I chose Recursive Character Splitting with a chunk size of 1000 and a 200-character overlap.

Why: This strategy is superior to simple splitting because it attempts to keep paragraphs and sentences together, preserving the semantic meaning. The overlap ensures that context isn't lost when a topic is split across two chunks.

Embedding Model:

Google Generative AI Embeddings (models/embedding-001).

Why: It offers high-dimensional accuracy and integrates seamlessly with the Gemini LLM for consistent performance across the pipeline.

Vector Database: 

FAISS (Facebook AI Similarity Search).

Why: FAISS was selected because it is lightweight, runs locally without a cloud subscription, and allows for disk persistence, meaning the documents only need to be indexed once.

## 🚀 Setup & Execution
Clone the Repository:

Bash

git clone https://github.com/Pranaynellutla/RAG-Document-QA-Bot.git
cd RAG-Document-QA-Bot

Install Requirements:

Bash

pip install -r requirements.txt

Environment Variables:

Create a .env file in the root directory:

Plaintext
GOOGLE_API_KEY=your_gemini_api_key_here
Run Ingestion (Indexing):

Bash
python ingest.py
Run the Bot:

Bash
streamlit run app.py

## ❓ Example Queries
"What is the impact of Bitcoin on financial systems?" (Expected theme: Decentralization and gold-like status).

"How does science influence law?" (Expected theme: Forensic evidence and legislative updates).

"Summarize the Teesside University course details." (Expected theme: Master's program modules and ROI).

"What are the AI internship requirements?" (Expected theme: RAG pipeline components and delivery dates).

"Is there information about Mars in these documents?" (Expected theme: The bot should state it cannot find this information).


## ⚠️ Known Limitations
Data Format: Currently only supports PDF files; TXT or DOCX files are not processed in the current version.

Static Index: If new files are added to the /data folder, the ingest.py script must be manually re-run to update the vector store.

API Dependency: Requires an active internet connection and a valid Google API key to function.
