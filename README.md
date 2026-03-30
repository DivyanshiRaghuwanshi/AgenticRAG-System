# Agentic RAG Assistant

Agentic RAG Assistant is an advanced AI system built to intelligently answer questions by combining document Q&A (RAG) with live web search. It combines document Q&A (RAG) with live web search and supports Groq, OpenAI, and Gemini in one interface.

Live Demo: https://agenticrag-system-ssgr9qksfpbm52omuekgu2.streamlit.app/

Video Recording: https://drive.google.com/file/d/1OOOm06JvsRTYyjCt3P4N0iTRR_286Yys/view?usp=sharing

## Why this project

Most assistants are good at either static knowledge or current events, not both. This project solves that gap by letting users query uploaded files and the web from the same chat session, with clear control over how responses are generated.

## Features

### 1. Document Q&A with RAG

- Upload PDF, TXT, DOCX, XLSX (Excel), and PPTX (PowerPoint) files
- Split documents into chunks with overlap for context continuity
- Create embeddings locally using all-MiniLM-L6-v2
- Store vectors in FAISS
- **Advanced Retrieval:** Uses an LLM-driven MultiQuery approach to generate 3 semantic variations of the user question, improving recall matching against FAISS.
- Generate grounded answers from retrieved chunks with source references

### 2. Live Web Search

- Uses Serper API for real-time web results
- Returns cleanly formatted sources and snippets
- Handles timeout and API errors with readable fallback messages

### 3. Concise and Detailed response modes

- Concise mode for quick, direct replies
- Detailed mode for structured, expanded answers
- Switch instantly from the sidebar

### 4. Search mode is user controlled

This is an important behavior in this project:

- Users explicitly choose which tools are available through sidebar toggles
- Document Search (RAG) can be enabled or disabled
- Web Search can be enabled or disabled
- The agent can only use tools that the user has enabled

So, tool access is under user control. The agent decides how to answer only within those selected limits.

### 5. Multi-provider LLM support

- Groq: llama-3.1-8b-instant
- OpenAI: gpt-4o-mini
- Gemini: gemini-2.5-flash
- Ollama: Self-hosted local fallback using llama3.2

Default provider in the current codebase: Gemini.

### 6. Session memory and context trimming

- Uses MemorySaver with per-session thread_id
- Keeps context manageable by trimming to recent messages
- Supports long conversations without unbounded prompt growth

### 7. MCP Server Integration

- Includes a standalone `mcp_server.py` implementation
- Decouples the backend RAG and Web tools as a Model Context Protocol (MCP) JSON-RPC STDIO server
- Can be plugged securely into modern external AI IDEs (like Cursor or Claude Desktop)

## How it works

1. User sends a query
2. App builds the toolset based on sidebar toggles
3. If RAG is enabled, the agent can call get_answer
4. If Web Search is enabled, the agent can call search_web
5. Retrieved evidence is synthesized into final user-facing output

## Project structure

```text
project/
|-- app.py
|-- mcp_server.py
|-- requirements.txt
|-- README.md
|
|-- config/
|   `-- config.py
|
|-- models/
|   |-- llm.py
|   `-- embeddings.py
|
|-- prompts/
|   |-- agent_prompt.py
|   `-- rag_prompt.py
|
|-- utils/
|   |-- rag_utils.py
|   |-- search_utils.py
|   |-- tools.py
|   `-- agent_utils.py
|
`-- .streamlit/
    `-- config.toml
```

## Tech stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| Agent orchestration | LangChain agent with tool calling |
| Memory/checkpointing | LangGraph MemorySaver |
| LLM integrations | langchain-openai, langchain-groq, langchain-google-genai |
| Embeddings | langchain-huggingface (all-MiniLM-L6-v2) |
| Vector store | FAISS |
| Document loaders | PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredExcelLoader, UnstructuredPowerPointLoader |
| Text splitting | RecursiveCharacterTextSplitter |
| Web search | Serper API via requests |

## Local setup

### Prerequisites

- Python 3.9+
- At least one LLM API key
- Serper API key for web search

### 1. Clone

```bash
git clone <your-repo-url>
cd project
```

### 2. Create and activate virtual environment

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

Create a .env file in the project root:

```env
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
SERPER_API_KEY=your_serper_key
OPENROUTER_API_KEY=your_openrouter_key
```

Do not commit keys to GitHub.

### 5. Run

```bash
streamlit run app.py
```

Then open http://localhost:8501.

## Using the App

1. **Select LLM Provider** from the sidebar (Gemini is the default).
2. **Choose Response Mode** — Concise for quick answers, Detailed for structured explanations.
3. **Search Mode** — Toggle Document Search (RAG) and Web Search on or off. The agent only uses tools you have enabled.
4. **Upload Documents** — Drag your PDFs, Word files, or text files into the upload box. You'll see a chunk count confirming the index was built.
5. **Start chatting** — Ask anything. The agent uses the tools you have enabled to find and compose an answer.

### Example Queries

**With documents uploaded:**
- "Summarize this document"
- "What projects are mentioned in the resume?"
- "Explain the Self-Supervised Methodology section in detail"
- "What are the key findings in chapter 3?"

**Web search:**
- "Latest developments in LLM agent frameworks"
- "What is the current state of RAG in production?"
- "Recent AI research on vector databases"

**Hybrid (both tools):**
- "What does the document say about transformers, and what are the latest improvements to that architecture?"

---

## Configuration Reference

All tunable parameters live in `config/config.py`:

| Parameter | Default | What it controls |
|---|---|---|
| `DEFAULT_LLM_PROVIDER` | `"gemini"` | Which provider loads on startup |
| `RETRIEVAL_TEMPERATURE` | `0.0` | LLM temperature for RAG extraction (factual) |
| `RESPONSE_TEMPERATURE` | `0.3` | LLM temperature for final agent response |
| `CHUNK_SIZE` | `1000` | Characters per document chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks for context continuity |
| `TOP_K` | `7` | Number of chunks retrieved per query |
| `MAX_HISTORY_MESSAGES` | `20` | Messages kept in context window per session |
| `SERPER_NUM_RESULTS` | `5` | Web search results returned per query |

---

## Deployment on Streamlit Cloud

### Step 1 — Push to GitHub

```bash
git add .
git commit -m "initial commit"
git push origin main
```

### Step 2 — Deploy

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Connect your GitHub account and select the repository
4. Set **Main file path** to `app.py`
5. Click **Advanced settings** and open the **Secrets** tab
6. Paste your API keys in TOML format:

```toml
GROQ_API_KEY    = "your_groq_key"
OPENAI_API_KEY  = "your_openai_key"
GEMINI_API_KEY  = "your_gemini_key"
SERPER_API_KEY  = "your_serper_key"
```

7. Click **Deploy**

Streamlit Cloud will install dependencies from `requirements.txt` and start the app. The HuggingFace model downloads automatically on first boot.

---

## Design Decisions

**Why LangGraph instead of a basic LangChain agent?**
LangGraph's `create_react_agent` with `MemorySaver` gives proper checkpoint-based memory tied to a `thread_id`. This means each user session is truly isolated — even in a shared deployment, two users won't see each other's conversation history. A basic LangChain agent with a message list doesn't give you this.

**Why local embeddings instead of OpenAI embeddings?**
`all-MiniLM-L6-v2` runs entirely on CPU with no API calls. This means document indexing is free regardless of file size, and there's no network dependency during the embedding step. For a demo and case study submission, this is the right call. A production deployment targeting scale would switch to a hosted embedding service.

**Why FAISS over a cloud vector database?**
FAISS is entirely in-memory, highly portable across operating systems, and has zero infrastructure overhead. While Milvus Lite is a powerful choice, it frequently triggers architecture compilation errors on Windows OS environments. FAISS guarantees maximum cross-platform compatibility. For a single-session demo, it's perfectly appropriate. The architecture is modular enough that swapping FAISS for Pinecone or Weaviate would only require changing `rag_utils.py` — the rest of the system doesn't care.

**Why two separate LLM instances at different temperatures?**
temperature=0 for fact extraction from documents, temperature=0.3 for conversational responses. Mixing these would compromise one or the other.

---

## Security Notes

- API keys are loaded from `.streamlit/secrets.toml` locally and from Streamlit Cloud Secrets in production
- `secrets.toml` is in `.gitignore` and will never appear in the repository
- No API keys are displayed anywhere in the UI
- User-uploaded documents are processed in memory and are not persisted to disk between sessions

---

## Troubleshooting

**App won't start / import errors**
Make sure you're running from inside the `project/` folder and your virtual environment is activated.
```bash
cd project
streamlit run app.py
```

**"No module named X" error**
```bash
pip install -r requirements.txt
```

**Groq XML tool-call error** (`tool_use_failed`)
This is a known issue with certain Groq llama models generating `<function=...>` format instead of JSON. Switch to OpenAI from the sidebar or use `llama-3.1-8b-instant` (most stable Groq option).

**Gemini quota errors**
Free-tier Gemini API keys from Google AI Studio may have zero quota on newer models. This requires a Google Cloud project with billing enabled. OpenAI or Groq are the recommended alternatives.

**Document upload shows 0 chunks**
The uploaded PDF may be image-based (scanned) rather than text-based. Text-based PDFs have extractable content. Image-based PDFs require OCR which is not included in this version.

**Port already in use**
Another Streamlit instance is running. Kill it first:
```bash
# Windows PowerShell
Get-Process python | Stop-Process -Force
streamlit run app.py
```
