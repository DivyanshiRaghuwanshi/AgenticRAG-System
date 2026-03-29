from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openrouter import ChatOpenRouter
from langchain_community.chat_models import ChatOllama
from config.config import (
    GROQ_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, OPEN_ROUTER_API_KEY,
    GROQ_MODEL_NAME, OPENAI_MODEL_NAME, GEMINI_MODEL_NAME, OPENROUTER_MODEL_NAME, OLLAMA_MODEL_NAME,
    RETRIEVAL_TEMPERATURE, RESPONSE_TEMPERATURE,
    DEFAULT_LLM_PROVIDER,
)


def get_llm(provider=None, temperature=RESPONSE_TEMPERATURE):
    provider = (provider or DEFAULT_LLM_PROVIDER).lower().strip()
    try:
        if provider == "groq":
            return ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL_NAME, temperature=temperature)
        elif provider == "openai":
            return ChatOpenAI(api_key=OPENAI_API_KEY, model=OPENAI_MODEL_NAME, temperature=temperature)
        elif provider == "gemini":
            return ChatGoogleGenerativeAI(google_api_key=GEMINI_API_KEY, model=GEMINI_MODEL_NAME, temperature=temperature)
        elif provider == "ollama":
            return ChatOllama(model=OLLAMA_MODEL_NAME, temperature=temperature)
        else:
            raise ValueError(f"Unknown provider '{provider}'. Choose: groq, openai, gemini, openrouter, ollama.")
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Could not initialize '{provider}' LLM: {e}")


def get_retrieval_llm(provider=None):
    # temperature=0 for factual extraction inside get_answer tool
    return get_llm(provider=provider, temperature=RETRIEVAL_TEMPERATURE)


def get_response_llm(provider=None):
    # temperature=0.3 for the main ReAct agent responses
    return get_llm(provider=provider, temperature=RESPONSE_TEMPERATURE)
