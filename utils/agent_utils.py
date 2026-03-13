import logging

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

from config.config import MAX_HISTORY_MESSAGES

logger = logging.getLogger(__name__)


def _make_trim_hook(max_messages=MAX_HISTORY_MESSAGES):
    # Middleware: trims old messages before each LLM call to avoid token limit errors
    def trim_hook(state):
        messages = state.get("messages", [])
        if len(messages) > max_messages:
            messages = messages[-max_messages:]
        return {"messages": messages}
    return trim_hook


def build_agent(response_llm, tools, system_prompt, memory):
    """Builds a LangGraph ReAct agent with InMemorySaver and message trimming."""
    try:
        return create_react_agent(
            model=response_llm,
            tools=tools,
            checkpointer=memory,
            prompt=system_prompt,
            pre_model_hook=_make_trim_hook(),
        )
    except Exception as e:
        raise RuntimeError(f"Failed to build agent: {e}")


def run_agent(agent, user_message, thread_id="default"):
    """Sends a message to the agent and returns the reply string."""
    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config={"configurable": {"thread_id": thread_id}},
        )
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                return msg.content
        return "I could not generate a response. Please try again."
    except Exception as e:
        logger.error(f"Agent error (thread={thread_id}): {e}")
        raise RuntimeError(f"Agent error: {e}")
