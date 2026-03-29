import os
import tempfile
import uuid
import logging

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from models.llm import get_retrieval_llm
from langchain_core.messages import SystemMessage, HumanMessage

from models.embeddings import get_embedding_model
from config.config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K

# Set up logging for the MultiQueryRetriever so we can see the generated queries
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


def _get_loader(file_path, file_ext):
    ext = file_ext.lower().strip(".")
    if ext == "pdf":
        return PyPDFLoader(file_path)
    elif ext == "txt":
        return TextLoader(file_path, encoding="utf-8")
    elif ext == "docx":
        return Docx2txtLoader(file_path)
    elif ext in ["xlsx", "xls"]:
        return UnstructuredExcelLoader(file_path)
    elif ext in ["pptx", "ppt"]:
        return UnstructuredPowerPointLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type '.{ext}'. Upload a PDF, TXT, DOCX, XLSX, or PPTX.")


def load_documents(file_path, file_ext):
    try:
        loader = _get_loader(file_path, file_ext)
        return loader.load()
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Error loading document: {e}")


def split_documents(documents):
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )
        return splitter.split_documents(documents)
    except Exception as e:
        raise RuntimeError(f"Error splitting documents: {e}")


def build_vectorstore(chunks):
    try:
        embedding_model = get_embedding_model()
        return FAISS.from_documents(chunks, embedding_model)
    except Exception as e:
        raise RuntimeError(f"Error building vector store: {e}")


def retrieve_relevant_chunks(query, vectorstore, k=TOP_K, llm=None):
    """
    Advanced Retrieval: Uses an LLM to generate 3 variations of the user's query,
    searches the vector DB for all of them, and returns the unique chunks.
    This significantly improves recall over standard KNN search.
    """
    try:
        if llm is None:
            llm = get_retrieval_llm()
        
        # Step 1: Generate multiple queries
        prompt = (
            "You are an AI assistant. Your task is to generate 3 different versions of the given user "
            "question to retrieve relevant documents from a vector database. Provide these alternative "
            "questions separated by newlines, with no numbering or extra text.\n"
            f"Original question: {query}"
        )
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            queries = response.content.strip().split("\n")
            queries = [q.strip() for q in queries if q.strip()]
        except Exception:
            # Fallback if LLM fails
            queries = []
            
        # Make sure the original query is always included
        queries.append(query)
        queries = list(set(queries))
        
        # Step 2: Search for all queries and combine unique chunks
        unique_chunks = {}
        for q in queries:
            results = vectorstore.similarity_search(q, k=k)
            for chunk in results:
                # Use page_content as a simple uniqueness key
                content_hash = hash(chunk.page_content)
                if content_hash not in unique_chunks:
                    unique_chunks[content_hash] = chunk
                    
        # Return the top chunks (we can return slightly more since we aggregated)
        return list(unique_chunks.values())[:k * 2]
        
    except Exception as e:
        raise RuntimeError(f"Error during advanced chunk retrieval: {e}")


def process_uploaded_file(uploaded_file):
    """Loads, chunks and indexes an uploaded file. Returns (vectorstore, chunk_count)."""
    file_ext = uploaded_file.name.rsplit(".", 1)[-1]
    tmp_file = None
    try:
        # Write to temp file so LangChain loaders can open it by path
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_file = tmp.name

        documents   = load_documents(tmp_file, file_ext)

        # Replace temp path with the real filename in metadata
        for doc in documents:
            doc.metadata["source"] = uploaded_file.name
            if "page" in doc.metadata:
                doc.metadata["page"] = doc.metadata["page"] + 1  # 1-indexed

        chunks      = split_documents(documents)
        vectorstore = build_vectorstore(chunks)

        return vectorstore, len(chunks)

    except Exception as e:
        raise RuntimeError(f"Failed to process '{uploaded_file.name}': {e}")

    finally:
        if tmp_file and os.path.exists(tmp_file):
            os.unlink(tmp_file)
