import os
import sys
import json
import logging
import asyncio
from typing import Dict, Any

# We mock our internal imports if run isolated
try:
    from utils.rag_utils import retrieve_relevant_chunks
    from utils.search_utils import web_search
    from config.config import TOP_K
except ImportError:
    pass # Can happen if paths are weird, but normally fine if run from root.

# Simple STDIO MCP Server Implementation
# (Implementing a minimal subset of the Model Context Protocol over stdout)

logging.basicConfig(level=logging.INFO, filename="mcp_server.log")
logger = logging.getLogger("mcp_server")

def mcp_respond(response_id: str, result: Any):
    msg = {
        "jsonrpc": "2.0",
        "id": response_id,
        "result": result
    }
    print(json.dumps(msg), flush=True)

def mcp_error(response_id: str, code: int, message: str):
    msg = {
        "jsonrpc": "2.0",
        "id": response_id,
        "error": {
            "code": code,
            "message": message
        }
    }
    print(json.dumps(msg), flush=True)

def handle_initialize(request: Dict):
    mcp_respond(request["id"], {
        "protocolVersion": "0.1.0",
        "serverInfo": {
            "name": "Agentic-RAG-Assistant-MCP",
            "version": "1.0.0"
        },
        "capabilities": {
            "tools": {}
        }
    })

def handle_tools_list(request: Dict):
    mcp_respond(request["id"], {
        "tools": [
            {
                "name": "web_search",
                "description": "Search the web for current, real-time information. Useful for recent events.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "knowledge_base_search",
                "description": "Search the internal uploaded document vector database.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]
    })

def handle_tool_call(request: Dict):
    params = request.get("params", {})
    tool_name = params.get("name")
    tool_args = params.get("arguments", {})
    query = tool_args.get("query", "")

    if tool_name == "web_search":
        try:
            # Import dynamically to ensure env is ready
            from utils.search_utils import web_search
            result = web_search(query)
            mcp_respond(request["id"], {"content": [{"type": "text", "text": result}]})
        except Exception as e:
            mcp_error(request["id"], -32000, str(e))
            
    elif tool_name == "knowledge_base_search":
        try:
            from config.config import TOP_K
            from models.embeddings import get_embedding_model
            from langchain_milvus import Milvus
            
            # Find the active Milvus DB
            embedding_model = get_embedding_model()
            # If the user has initialized a DB in the UI, we try to load it
            if os.path.exists("./milvus_local.db"):
                vs = Milvus(embedding_model, connection_args={"uri": "./milvus_local.db"}, collection_name="mcp_read")
                # Fallback to simple similarity search since MCP doesn't load the LLM multi-query dynamically easily
                chunks = vs.similarity_search(query, k=TOP_K)
                res = "\n".join([c.page_content for c in chunks])
                mcp_respond(request["id"], {"content": [{"type": "text", "text": res if res else "No results."}]})
            else:
                 mcp_respond(request["id"], {"content": [{"type": "text", "text": "No local document database found. Please upload documents in UI."}]})
        except Exception as e:
            mcp_error(request["id"], -32000, str(e))
    else:
        mcp_error(request["id"], -32601, f"Tool {tool_name} not found")

async def main():
    """Main loop for reading MCP JSON-RPC messages from stdin."""
    logger.info("Starting ResearchIQ MCP Server...")
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        
        try:
            request = json.loads(line)
            method = request.get("method")
            
            if method == "initialize":
                handle_initialize(request)
            elif method == "tools/list":
                handle_tools_list(request)
            elif method == "tools/call":
                handle_tool_call(request)
            else:
                mcp_error(request.get("id", "none"), -32601, f"Method {method} not found")
        except json.JSONDecodeError:
            continue
        except Exception as e:
            logger.error(f"Error handling request: {e}")

if __name__ == "__main__":
    asyncio.run(main())