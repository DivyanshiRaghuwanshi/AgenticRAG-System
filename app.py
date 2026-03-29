import streamlit as st

st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

import uuid
import logging

from langgraph.checkpoint.memory import MemorySaver
import config.config as cfg
from models.llm import get_retrieval_llm, get_response_llm
from utils.rag_utils import process_uploaded_file
from utils.tools import create_get_answer_tool, create_search_web_tool
from utils.agent_utils import build_agent, run_agent_stream
from prompts.agent_prompt import get_agent_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Custom CSS for styling
st.markdown(
    """
    <style>
        .main { padding-top: 1rem; }

        section[data-testid="stSidebar"] {
            background-color: #0f1117;
            border-right: 1px solid #2a2d3e;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.5rem;
        }

        .app-header {
            text-align: center;
            padding: 1.2rem 0 0.5rem 0;
        }
        .app-header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(90deg, #4F8BF9, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
        }
        .app-header p {
            color: #8b92a5;
            font-size: 0.95rem;
            margin-top: 0.25rem;
        }

        .status-ok   { color: #4ade80; font-weight: 600; }
        .status-warn { color: #facc15; font-weight: 600; }
        .status-err  { color: #f87171; font-weight: 600; }

        hr { border-color: #2a2d3e; }

        .info-box {
            background: #1a1d2e;
            border: 1px solid #2a2d3e;
            border-radius: 8px;
            padding: 1rem 1.25rem;
            margin-bottom: 1rem;
            color: #8b92a5;
            font-size: 0.88rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# Initialize session state on first load
def _init_session_state():
    defaults = {
        "messages":     [],
        "thread_id":    str(uuid.uuid4()),
        "vectorstore":  None,
        "agent":        None,
        "memory":       MemorySaver(),
        "doc_names":    [],
        "agent_config": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session_state()


def _agent_fingerprint(provider, response_mode, doc_loaded, use_rag, use_web):
    model = {"groq": cfg.GROQ_MODEL_NAME, "openai": cfg.OPENAI_MODEL_NAME, "gemini": cfg.GEMINI_MODEL_NAME}.get(provider, "")
    return f"{provider}|{model}|{response_mode}|{doc_loaded}|{use_rag}|{use_web}"


def _build_or_rebuild_agent(provider, response_mode, use_rag, use_web):
    """Builds the agent, or rebuilds it if provider/mode/docs/tools changed."""
    doc_loaded  = st.session_state.vectorstore is not None
    fingerprint = _agent_fingerprint(provider, response_mode, doc_loaded, use_rag, use_web)

    if st.session_state.agent and st.session_state.agent_config == fingerprint:
        return  # nothing changed, reuse existing agent

    try:
        retrieval_llm = get_retrieval_llm(provider)
        response_llm  = get_response_llm(provider)

        tools = []
        if use_rag:
            tools.append(create_get_answer_tool(
                vectorstore=st.session_state.vectorstore,
                retrieval_llm=retrieval_llm,
            ))
        if use_web:
            tools.append(create_search_web_tool())

        if not tools:
            st.warning("Enable at least one search mode to get answers.")
            return

        agent = build_agent(
            response_llm=response_llm,
            tools=tools,
            system_prompt=get_agent_prompt(response_mode, use_rag, use_web),
            memory=st.session_state.memory,
        )

        st.session_state.agent        = agent
        st.session_state.agent_config = fingerprint

    except Exception as e:
        st.error(f"Could not initialise the agent: {e}")
        st.session_state.agent = None


# Sidebar
with st.sidebar:

    st.markdown("## 🔍 RAG Assistant")
    st.markdown(
        "<p style='color:#8b92a5;font-size:0.85rem;margin-top:-0.5rem;'>"
        "Document + Web Research Assistant</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # LLM provider selection
    provider_options = ["groq", "openai", "gemini", "ollama"]
    provider = st.selectbox(
        "LLM Provider",
        options=provider_options,
        index=provider_options.index(cfg.DEFAULT_LLM_PROVIDER) if cfg.DEFAULT_LLM_PROVIDER in provider_options else 0,
        help="Choose which model powers the assistant.",
    )

    st.divider()

    # Response mode toggle
    st.markdown("**Response Mode**")
    response_mode_label = st.radio(
        "Response Mode",
        options=["Concise", "Detailed"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        help="Concise: short answers. Detailed: in-depth explanations.",
    )
    response_mode = response_mode_label.lower()

    st.divider()

    # Search mode toggles
    st.markdown("**Search Mode**")
    use_rag = st.checkbox("Document Search (RAG)", value=True, help="Search your uploaded documents for answers.")
    use_web = st.checkbox("Web Search", value=True, help="Search the web for current or real-time information.")

    st.divider()
    st.markdown("**Upload Knowledge Base**")
    st.caption("PDF, TXT, DOCX, XLSX, or PPTX. Multiple files supported.")

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "txt", "docx", "xlsx", "xls", "pptx", "ppt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        new_names = sorted([f.name for f in uploaded_files])
        if new_names != st.session_state.doc_names:
            with st.spinner("Indexing documents…"):
                try:
                    combined_vs  = None
                    total_chunks = 0

                    for uploaded_file in uploaded_files:
                        vs, n_chunks = process_uploaded_file(uploaded_file)
                        total_chunks += n_chunks
                        if combined_vs is None:
                            combined_vs = vs
                        else:
                            combined_vs.merge_from(vs)

                    st.session_state.vectorstore  = combined_vs
                    st.session_state.doc_names    = new_names
                    st.session_state.agent_config = ""  # force agent rebuild

                    st.success(f"Indexed {len(uploaded_files)} file(s) → {total_chunks} chunks")

                except Exception as e:
                    st.error(f"Upload failed: {e}")

    if st.session_state.doc_names:
        st.markdown("**Loaded documents:**")
        for name in st.session_state.doc_names:
            st.markdown(f"<span class='status-ok'>✔</span> {name}", unsafe_allow_html=True)
    else:
        st.markdown(
            "<span class='status-warn'>⚠ No documents loaded — web search only mode</span>",
            unsafe_allow_html=True,
        )

    st.divider()

    # Status indicators
    st.markdown("**Status**")
    llm_ok    = bool(
        (provider == "groq"   and cfg.GROQ_API_KEY)  or
        (provider == "openai" and cfg.OPENAI_API_KEY) or
        (provider == "gemini" and cfg.GEMINI_API_KEY) or
        (provider == "ollama") # Ollama runs locally, no API key required
    )
    search_ok = bool(cfg.SERPER_API_KEY)
    kb_ok     = st.session_state.vectorstore is not None

    llm_icon    = "✔" if llm_ok    else "✖"
    kb_icon     = "✔" if kb_ok     else "–"
    
    if not use_web:
        search_icon = "–"
        search_class = "status-warn"
        search_label = "Web Search (Disabled)"
    elif not search_ok:
        search_icon = "✖"
        search_class = "status-err"
        search_label = "Web Search (Missing Key)"
    else:
        search_icon = "✔"
        search_class = "status-ok"
        search_label = "Web Search (Serper)"

    llm_class    = "status-ok" if llm_ok    else "status-err"
    kb_class     = "status-ok" if kb_ok     else "status-warn"

    st.markdown(
        f"<span class='{llm_class}'>{llm_icon} LLM ({provider})</span><br>"
        f"<span class='{search_class}'>{search_icon} {search_label}</span><br>"
        f"<span class='{kb_class}'>{kb_icon} Knowledge Base</span>",
        unsafe_allow_html=True,
    )

    st.divider()

    if st.button("🗑  Clear Conversation", use_container_width=True):
        st.session_state.messages     = []
        st.session_state.thread_id    = str(uuid.uuid4())
        st.session_state.memory       = MemorySaver()
        st.session_state.agent        = None
        st.session_state.agent_config = ""
        st.rerun()


# Main chat area
st.markdown(
    "<div class='app-header'><h1>Agentic RAG Assistant</h1>"
    "<p>Ask questions from your documents or let me search the web for you.</p></div>",
    unsafe_allow_html=True,
)

# Welcome message shown before any chat
if not st.session_state.messages:
    st.markdown(
        """
        <div class='info-box'>
        <b>Getting started</b><br><br>
        1. Upload one or more documents (PDF, TXT, DOCX) to build a knowledge base.<br>
        2. Choose <b>Concise</b> for quick answers or <b>Detailed</b> for deep explanations.<br>
        3. Ask anything — the assistant searches your documents, the web, or both.<br><br>
        <i>No documents uploaded? Web search mode is always on.</i>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask a question about your documents or any topic…")

if user_input:
    if not llm_ok:
        st.warning(f"Please add a valid API key for **{provider}** to start chatting.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    _build_or_rebuild_agent(provider, response_mode, use_rag, use_web)

    if st.session_state.agent is None:
        with st.chat_message("assistant"):
            st.error("Agent could not be initialised. Check your API key and try again.")
        st.stop()

    with st.chat_message("assistant"):
        status_container = st.status("Agent is thinking...", expanded=True)
        final_answer = ""
        try:
            # We iterate over the stream to show the thought process
            for messages in run_agent_stream(
                agent=st.session_state.agent,
                user_message=user_input,
                thread_id=st.session_state.thread_id
            ):
                # Look at the most recently added message
                last_msg = messages[-1]
                
                # If it's an AI message, it might be calling a tool
                if last_msg.type == "ai":
                    if last_msg.tool_calls:
                        for tool_call in last_msg.tool_calls:
                            tool_name = tool_call["name"]
                            status_container.write(f"🧠 Agent decided to use tool: `{tool_name}`...")
                    elif last_msg.content:
                        # This is the final text response normally
                        status_container.write("✅ Agent formulated the final response.")
                        
                        # Handle varied response formats (some LLMs like Gemini return lists of dictionaries)
                        content = last_msg.content
                        if isinstance(content, list):
                            text_parts = []
                            for part in content:
                                if isinstance(part, dict) and "text" in part:
                                    text_parts.append(part["text"])
                                elif isinstance(part, str):
                                    text_parts.append(part)
                            final_answer = "\n".join(text_parts)
                        else:
                            final_answer = str(content)

                # If it's a Tool message, it means the tool finished executing
                elif last_msg.type == "tool":
                    status_container.write(f"🔍 Tool `{last_msg.name}` returned results.")
            
            status_container.update(label="Response generated", state="complete", expanded=False)
            
            # Display the final answer outside the status container
            st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})

        except Exception as e:
            status_container.update(label="Error occurred", state="error")
            st.error(f"Something went wrong: {e}")
            logger.error(f"Agent error: {e}", exc_info=True)
