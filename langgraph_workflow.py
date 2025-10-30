from typing import TypedDict, Annotated, Sequence
import operator
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM
from rag import build_rag_chain
from vector import get_vectorstores

# ------------------- Load all vectorstores -------------------
vectorstores = get_vectorstores()

heart_rag, heart_retriever = build_rag_chain(vectorstores["heart"])
gyno_rag, gyno_retriever = build_rag_chain(vectorstores["gyno"])

model = OllamaLLM(model="llama3", temperature=0.3)

# ------------------- State -------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[str], operator.add]
    route: str  # HEART / GYNO / LLM

# ------------------- Supervisor Node -------------------
def supervisor_node(state: AgentState):
    question = state["messages"][-1]

    prompt = f""" You are a supervisor node in a medical AI workflow.
Your task is to analyze the user's question and assign it to the correct domain based on its content and intent.
Choose exactly one category from the following:

HEART — if the question is related to:
Cardiology, ECG, blood pressure, chest pain, cholesterol, heart rate, heart diseases, cardiac care, or cardiologist consultations.

GYNO — if the question is related to:
Gynecology, menstruation, pregnancy, fertility, PCOS, contraception, reproductive or sexual health, female wellness, or any question about Kokoro.Doctor’s website, doctors, appointments, booking, or profiles.

LLM — if the question is:
General, unrelated to medical topics, or about AI/LLM/chatbot behavior, coding, or other non-medical subjects.

Respond with only one word — exactly one of: HEART, GYNO, or LLM.
Question: {question}
"""
    response = model.invoke(prompt).strip().upper()
    if "HEART" in response:
        route = "HEART"
    elif "GYNO" in response:
        route = "GYNO"
    else:
        route = "LLM"
    return {"messages": [question], "route": route}

# ------------------- RAG Nodes -------------------
def heart_rag_node(state: AgentState):
    question = state["messages"][-1]
    docs = heart_retriever.invoke(question)
    if not docs:
        return {"messages": ["No relevant cardiac info found."]}
    context = "\n\n".join([d.page_content for d in docs])
    response = heart_rag.invoke(question)
    return {"messages": [response]}

def gyno_rag_node(state: AgentState):
    question = state["messages"][-1]
    docs = gyno_retriever.invoke(question)
    if not docs:
        return {"messages": ["No relevant gynecology info found."]}
    context = "\n\n".join([d.page_content for d in docs])
    response = gyno_rag.invoke(question)
    return {"messages": [response]}

# ------------------- LLM Node -------------------
def llm_node(state: AgentState):
    question = state["messages"][-1]
    response = model.invoke(f"Answer this in a helpful way: {question}")
    return {"messages": [response]}

# ------------------- Router -------------------
def router(state: AgentState):
    route = state.get("route", "LLM")
    if route == "HEART":
        return "heart_rag_node"
    elif route == "GYNO":
        return "gyno_rag_node"
    else:
        return "llm_node"

# ------------------- Workflow -------------------
workflow_graph = StateGraph(AgentState)
workflow_graph.add_node("Supervisor", supervisor_node)
workflow_graph.add_node("heart_rag_node", heart_rag_node)
workflow_graph.add_node("gyno_rag_node", gyno_rag_node)
workflow_graph.add_node("llm_node", llm_node)

workflow_graph.set_entry_point("Supervisor")
workflow_graph.add_conditional_edges(
    "Supervisor", router,
    {"heart_rag_node": "heart_rag_node", "gyno_rag_node": "gyno_rag_node", "llm_node": "llm_node"}
)
workflow_graph.add_edge("heart_rag_node", END)
workflow_graph.add_edge("gyno_rag_node", END)
workflow_graph.add_edge("llm_node", END)

compiled_workflow = workflow_graph.compile()

# ------------------- Runner -------------------
def run_rag_pipeline(message: str, language: str = "en"):
    state = {"messages": [message], "route": ""}
    final_state = compiled_workflow.invoke(state)
    return final_state["messages"][-1]
