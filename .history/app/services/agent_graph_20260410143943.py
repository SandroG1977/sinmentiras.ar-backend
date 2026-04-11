from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from app.core.config import settings


class AgentState(TypedDict):
    prompt: str
    output: str


def _build_graph():
    def run_model(state: AgentState) -> AgentState:
        prompt = state["prompt"]

        # Fallback to a deterministic mock response when no API key is set.
        if not settings.OPENAI_API_KEY:
            return {"prompt": prompt, "output": f"[mock-response] {prompt}"}

        llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )
        response = llm.invoke(prompt)
        content = response.content if isinstance(response.content, str) else str(response.content)
        return {"prompt": prompt, "output": content}

    graph = StateGraph(AgentState)
    graph.add_node("run_model", run_model)
    graph.add_edge(START, "run_model")
    graph.add_edge("run_model", END)
    return graph.compile()


agent_graph = _build_graph()


def run_agent(prompt: str) -> tuple[str, str]:
    result = agent_graph.invoke({"prompt": prompt, "output": ""})
    source = settings.OPENAI_MODEL if settings.OPENAI_API_KEY else "mock"
    return result["output"], source
