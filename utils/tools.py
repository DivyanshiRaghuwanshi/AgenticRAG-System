"""
utils/tools.py

Defines the custom LangChain tools passed to the ReAct agent.

  1. create_get_answer_tool()  → RAG tool that searches the knowledge base
  2. create_search_web_tool()  → Live web search tool via Serper

Both are factory functions so we can inject runtime dependencies (the vector
store and retrieval LLM) without using global state or singletons.

NOTE on docstring length:
  Groq's llama models fail with XML-style tool calls when tool docstrings
  contain multi-line descriptions or 'Args:' sections. Keep descriptions
  short and single-line to stay within Groq's tool-call format expectations.
"""

from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

from utils.rag_utils import retrieve_relevant_chunks
from utils.search_utils import web_search
from config.config import TOP_K
from prompts.rag_prompt import RAG_AGENT_PROMPT


def create_get_answer_tool(vectorstore, retrieval_llm):
    """
    Factory: returns the get_answer tool with vectorstore and LLM injected.

    Dual-LLM RAG pipeline:
      Step 1 → Advanced MultiQuery search retrieves the most relevant chunks
      Step 2 → retrieval_llm (temperature=0) extracts a grounded answer from
               those chunks using RAG_AGENT_PROMPT as its system instruction
    """

    @tool
    def get_answer(query: str) -> str:
        """Search the uploaded knowledge base documents and return a cited answer."""
        try:
            if vectorstore is None:
                return "No knowledge base loaded. Please upload a document first."

            # Step 1: Advanced retrieval — fetch relevant chunks using MultiQuery
            chunks = retrieve_relevant_chunks(query, vectorstore, k=TOP_K, llm=retrieval_llm)

            if not chunks:
                return "No relevant information found in the uploaded documents."

            # Format chunks with source metadata for citation
            context_blocks = []
            for i, chunk in enumerate(chunks, start=1):
                source   = chunk.metadata.get("source", "Unknown document")
                page     = chunk.metadata.get("page", None)
                page_tag = f", page {page}" if page is not None else ""
                context_blocks.append(
                    f"[Source {i}: {source}{page_tag}]\n{chunk.page_content}"
                )
            context = "\n\n---\n\n".join(context_blocks)

            # Step 2: retrieval LLM (temp=0) synthesises a grounded answer
            messages = [
                SystemMessage(content=RAG_AGENT_PROMPT),
                HumanMessage(
                    content=(
                        f"Retrieved document chunks:\n\n{context}\n\n"
                        f"Question: {query}\n\n"
                        "Answer strictly from the retrieved chunks. "
                        "Cite the source numbers where relevant."
                    )
                ),
            ]
            response = retrieval_llm.invoke(messages)
            return response.content

        except Exception as e:
            return f"Error searching the knowledge base: {e}"

    return get_answer


def create_search_web_tool():
    """Factory: returns the search_web tool for live Google search via Serper."""

    @tool
    def search_web(query: str) -> str:
        """Search the web for current, real-time information on the given topic."""
        try:
            return web_search(query)
        except Exception as e:
            return f"Web search error: {e}"

    return search_web
