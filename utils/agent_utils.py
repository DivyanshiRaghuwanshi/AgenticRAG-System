import logging

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

from config.config import MAX_HISTORY_MESSAGES

logger = logging.getLogger(__name__)

def build_agent(response_llm, tools, system_prompt, memory):
    """Builds a LangGraph ReAct agent with InMemorySaver."""
    try:
        def state_modifier(state):
            # Message trimming to avoid token limits
            messages = state.get("messages", [])
            if len(messages) > MAX_HISTORY_MESSAGES:
                # Keep system prompt if it's the first message, and the most recent N messages
                if len(messages) > 0 and messages[0].type == "system":
                    state["messages"] = [messages[0]] + messages[-(MAX_HISTORY_MESSAGES-1):]
                else:
                    state["messages"] = messages[-MAX_HISTORY_MESSAGES:]
            return state

        return create_react_agent(
            model=response_llm,
            tools=tools,
            checkpointer=memory,
            prompt=system_prompt, # pass prompt instead of state_modifier
        )
    except Exception as e:
        raise RuntimeError(f"Failed to build agent: {e}")

def run_agent_stream(agent, user_message, thread_id="default"):
    """Yields events from the agent so the UI can show the thinking process."""
    try:
        config = {"configurable": {"thread_id": thread_id}}
        inputs = {"messages": [("user", user_message)]}
        
        # Stream the agent's work
        for event in agent.stream(inputs, config=config, stream_mode="values"):
            # We yield the whole list of messages every time the state changes
            if "messages" in event:
                yield event["messages"]
    except Exception as e:
        logger.error(f"Agent error (thread={thread_id}): {e}")
        raise RuntimeError(f"Agent error: {e}")

