import sys
import os
import json
from pathlib import Path
from typing import Annotated, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Modern LCEL imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Vector DB & models
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Configuration ---
FILE_PATH = './attacking_scenarios/enterprise-attack.json'
JQ_SCHEMA = '.objects[] | select(.type | contains(\"attack-pattern\"))'
PERSIST_DIRECTORY = './mitre_vector_db'

LLM_MODEL_NAME = ""

TOP_K_RETRIEVAL = 5
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Our RAG pipeline (LCEL)

app = FastAPI(
    title="GPT4All RAG Wrapper",
    description="Local RAG pipeline using HuggingFace GPU embeddings + GPT4All LLM.",
    version="1.0.0"
)


rag_chain: Optional[RunnableParallel] = None



def setup_vector_store(file_path: str, jq_schema: str, persist_directory: str) -> Chroma:
    persist_path = Path(persist_directory)

    # GPU-friendly HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda"}      # forces GPU use on RTX 3050
    )

    # If vector DB exists, reuse it
    if persist_path.exists() and any(persist_path.iterdir()):
        print("Loading existing Chroma DB (no re-embedding)...")
        return Chroma(
            embedding_function=embeddings,
            persist_directory=persist_directory
        )

    print("--- Loading JSON and creating chunks ---")

    loader = JSONLoader(
        file_path=file_path,
        jq_schema=jq_schema,
        text_content=False,
        json_lines=False
    )
    data = loader.load()

    # Improve RAG quality by including technique names
    for doc in data:
        name = doc.metadata.get("name", "N/A")
        description = doc.page_content
        doc.page_content = f"MITRE ATT&CK Technique: {name}\nDescription: {description}"

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    docs = text_splitter.split_documents(data)

    print(f"Loaded {len(data)} MITRE techniques → split into {len(docs)} chunks.")

    print("--- Building vector DB ---")

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    vectorstore.persist()
    print("Vector store created & saved.")
    return vectorstore



def build_rag_chain():
    print('--- Starting RAG Chain Initialization ---')
    try:
        # 1. Setup the Vector Store (Data Ingestion)
        # This function runs synchronously during startup, which is typical for initializing resources.
        vectorstore = setup_vector_store(FILE_PATH, JQ_SCHEMA, PERSIST_DIRECTORY)

        # 2. Initialize the Local LLM (qwen2-1_5b-instruct-q4_0)
        print(f"\nLoading LLM: {LLM_MODEL_NAME}")
        print(f"Loading GGUF model from: {LLM_MODEL_NAME}")
        llm = LlamaCpp(
            model_path=LLM_MODEL_NAME,
            n_ctx=4096,
            temperature=0.3,
            n_gpu_layers=-1,  # offload all layers to GPU (if compiled with CUDA)
            # n_threads=8,
            # n_batch=128,
        )
        print("LLM loaded successfully.")

        # 3. Set up the Retriever and RAG Chain
        retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RETRIEVAL})
        # 4. Build a modern RAG chain (no langchain.chains)
        system_prompt = (
            "You are a cybersecurity assistant specialized in the MITRE ATT&CK framework. "
            "Use the given system components to create 5 attack scenarios based on MITRE ATT&CK. "
            "System components:\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt)
            ]
        )
        retrieval_chain = (
            {"context": retriever}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        rag_chain = RunnableParallel(
            answer=retrieval_chain,
            source_documents=retriever,
        )
        print("RAG RetrievalQA chain created and ready.")
        return rag_chain
    except Exception as e:
        print(f"Error during RAG chain initialization: {e}", file=sys.stderr)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during RAG chain initialization: {e}. Check model and data files."
        )
        return None


class QueryRequest(BaseModel):
    prompt: Annotated[str, Field(
        description="The text prompt to send to the RAG chain",
        examples=["What is the MITRE technique for 'Phishing'?"]
    )]
    # We can keep these parameters, but they are often ignored by LangChain RAG chain directly
    # They are kept for API consistency but their function might need integration into the LLM call kwargs
    max_tokens: int = Field(2048, description="Maximum number of tokens to generate")
    temp: float = Field(0.7, description="Sampling temperature for creativity")

app = FastAPI(
    title="GPT4All RAG Wrapper",
    description="A FastAPI wrapper for a local GPT4All LLM using LangChain RAG pipeline with custom data.",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    global rag_chain
    rag_chain = build_rag_chain()


@app.post("/query/")
async def execute_prompt(query: QueryRequest):
    global rag_chain
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG Chain not initialized yet.")
    
    print(f"Executing RAG query: {query.prompt}")
    
    # New chain expects plain question string
    result = rag_chain.invoke(query.prompt)
    
    # Extract the generated response and source documents
    response = result.get("answer", "No answer generated.")
    
    sources = []
    for doc in result.get("source_documents", []):
        sources.append({
            "content_snippet": doc.page_content[:200] + "...",
            "metadata": doc.metadata
        })

    print("Response generated.")
    return {
        "prompt": query.prompt,
        "response": response,
        "source_documents": sources
    }
