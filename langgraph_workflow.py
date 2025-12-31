import logging
import operator
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM
from rag import build_rag_chain
from vector import get_vectorstores

# ------------------- 0. Logging Setup -------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("MedicalBot")

# ------------------- 1. Load Resources -------------------
logger.info("🔄 Loading Vector Stores...")
vectorstores = get_vectorstores()
heart_store = vectorstores["heart"]
gyno_store = vectorstores["gyno"] # Platform vectors are here
logger.info("✅ Vector Stores Loaded.")

model = OllamaLLM(model="llama3", temperature=0.1) # Lower temp for precision

# ------------------- 2. SYSTEM PROMPTS (PERSONAS) -------------------

# PATIENT: Friendly, safe, simple
PATIENT_SYS_PROMPT = """You are Kokoro, a helpful and empathetic medical health assistant for patients.
TONE: Warm, Reassuring, Simple, Clear.
INSTRUCTIONS:
- Explain medical concepts in plain language (no complex jargon without explanation).
- Be empathetic to pain or worry.
- If asked about the platform/app (Kokoro), answer based on the context provided.
"""

# DOCTOR: Professional, Clinical, Strict
DOCTOR_SYS_PROMPT = """You are Kokoro.Doctor, a clinical decision support assistant.
TONE: Formal, Objective, Precise, Professional.
INSTRUCTIONS:
- Use correct medical terminology, standard guidelines, and differential diagnoses.
- Provide answers in structured formats (bullet points, clinical notes).
- If asked about the platform/app, describe technical features or workflows based on context.

CRITICAL NEGATIVE CONSTRAINTS (DO NOT VIOLATE):
- NEVER use terms of endearment like 'sweetie', 'honey', 'dear', 'sweetheart', 'babe', or 'darling'.
- DO NOT be chatty, emotional, or overly casual.
- DO NOT hallucinate treatments not present in the context.
"""

# ------------------- 3. Build Chains -------------------
# Patient Chains
pt_heart_rag, pt_heart_retriever = build_rag_chain(heart_store, PATIENT_SYS_PROMPT)
pt_gyno_rag, pt_gyno_retriever = build_rag_chain(gyno_store, PATIENT_SYS_PROMPT)

# Doctor Chains
dr_heart_rag, dr_heart_retriever = build_rag_chain(heart_store, DOCTOR_SYS_PROMPT)
dr_gyno_rag, dr_gyno_retriever = build_rag_chain(gyno_store, DOCTOR_SYS_PROMPT)

# ------------------- 4. State Definition -------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[str], operator.add]
    role: str       
    next_node: str  

# ------------------- 5. ENTRY ROUTER (Role Split) -------------------
def initial_role_router(state: AgentState):
    role = state.get("role", "patient").lower()
    logger.info(f"🚦 START: Incoming request. Role detected: [{role.upper()}]")
    
    if role == "doctor":
        return "Doctor_Manager"
    else:
        return "Patient_Manager"

# ------------------- 6. PATIENT FLOW NODES -------------------

def patient_manager_node(state: AgentState):
    question = state["messages"][-1]
    logger.info(f"👤 Patient Manager: Analyzing question -> '{question}'")

    # Routing Logic: Includes 'Kokoro', 'You', 'Platform' in GYNO
    prompt = f"""You are a triage assistant for a Patient.
    Classify the question into: HEART, GYNO, or GENERAL.
    
    1. HEART: Chest pain, BP, cholesterol, cardiology, palpitations.
    2. GYNO (Includes Platform Info): Periods, pregnancy, sexual health AND Questions about 'Kokoro', 'Kokoro.Doctor', 'the app', 'you', 'your features', 'subscription', 'booking'.
    3. GENERAL: Greetings, coding, jokes, non-medical.
    
    Respond with one word only: HEART, GYNO, or GENERAL.
    Question: {question}"""
    
    response = model.invoke(prompt).strip().upper()
    logger.info(f"🧠 Classification (Patient): {response}")
    
    if "HEART" in response:
        return {"next_node": "pt_heart"}
    elif "GYNO" in response:
        return {"next_node": "pt_gyno"}
    else:
        return {"next_node": "pt_llm"}

def pt_heart_node(state: AgentState):
    res = pt_heart_rag.invoke(state["messages"][-1])
    return {"messages": [res]}

def pt_gyno_node(state: AgentState):
    # This node handles Gyno + Platform questions
    res = pt_gyno_rag.invoke(state["messages"][-1])
    return {"messages": [res]}

def pt_llm_node(state: AgentState):
    res = model.invoke(f"Speak like a helpful friend (Kokoro). User: {state['messages'][-1]}")
    return {"messages": [res]}

def patient_router(state: AgentState):
    route = state.get("next_node")
    if route == "pt_heart": return "pt_heart_node"
    if route == "pt_gyno": return "pt_gyno_node"
    return "pt_llm_node"

# ------------------- 7. DOCTOR FLOW NODES -------------------

def doctor_manager_node(state: AgentState):
    question = state["messages"][-1]
    logger.info(f"👨‍⚕️ Doctor Manager: Analyzing question -> '{question}'")

    # Routing Logic: Includes 'Kokoro', 'You', 'Platform' in GYNO
    prompt = f"""You are a triage assistant for a Doctor.
    Classify the question into: HEART, GYNO, or GENERAL.
    
    1. HEART: Cardiology, ECG, Angina, Beta-blockers, Heart Failure.
    2. GYNO (Includes Platform Info): OBGYN, Surgery, IVF AND Questions about 'Kokoro.Doctor', 'Kokoro', 'the platform', 'you', 'system capabilities', 'your features'.
    3. GENERAL: Casual chat, unrelated topics.
    
    Respond with one word only: HEART, GYNO, or GENERAL.
    Question: {question}"""
    
    response = model.invoke(prompt).strip().upper()
    logger.info(f"🧠 Classification (Doctor): {response}")
    
    if "HEART" in response:
        return {"next_node": "dr_heart"}
    elif "GYNO" in response:
        return {"next_node": "dr_gyno"}
    else:
        return {"next_node": "dr_llm"}

def dr_heart_node(state: AgentState):
    res = dr_heart_rag.invoke(state["messages"][-1])
    return {"messages": [f"**Clinical Response:**\n{res}"]}

def dr_gyno_node(state: AgentState):
    # This node handles Gyno + Platform questions for Doctors
    res = dr_gyno_rag.invoke(state["messages"][-1])
    return {"messages": [f"**Response:**\n{res}"]}

def dr_llm_node(state: AgentState):
    res = model.invoke(f"Speak like a professional medical colleague. Be concise. User: {state['messages'][-1]}")
    return {"messages": [res]}

def doctor_router(state: AgentState):
    route = state.get("next_node")
    if route == "dr_heart": return "dr_heart_node"
    if route == "dr_gyno": return "dr_gyno_node"
    return "dr_llm_node"


# ------------------- 8. GRAPH CONSTRUCTION -------------------
workflow = StateGraph(AgentState)

# Nodes
workflow.add_node("Patient_Manager", patient_manager_node)
workflow.add_node("Doctor_Manager", doctor_manager_node)
workflow.add_node("pt_heart_node", pt_heart_node)
workflow.add_node("pt_gyno_node", pt_gyno_node)
workflow.add_node("pt_llm_node", pt_llm_node)
workflow.add_node("dr_heart_node", dr_heart_node)
workflow.add_node("dr_gyno_node", dr_gyno_node)
workflow.add_node("dr_llm_node", dr_llm_node)

# Conditional Entry
workflow.set_conditional_entry_point(
    initial_role_router,
    {"Patient_Manager": "Patient_Manager", "Doctor_Manager": "Doctor_Manager"}
)

# Patient Edges
workflow.add_conditional_edges(
    "Patient_Manager",
    patient_router,
    {"pt_heart_node": "pt_heart_node", "pt_gyno_node": "pt_gyno_node", "pt_llm_node": "pt_llm_node"}
)

# Doctor Edges
workflow.add_conditional_edges(
    "Doctor_Manager",
    doctor_router,
    {"dr_heart_node": "dr_heart_node", "dr_gyno_node": "dr_gyno_node", "dr_llm_node": "dr_llm_node"}
)

# End Edges
for node in ["pt_heart_node", "pt_gyno_node", "pt_llm_node", "dr_heart_node", "dr_gyno_node", "dr_llm_node"]:
    workflow.add_edge(node, END)

compiled_workflow = workflow.compile()

def run_rag_pipeline(message: str, role: str = "patient"):
    state = {"messages": [message], "role": role, "next_node": ""}
    final_state = compiled_workflow.invoke(state)
    return final_state["messages"][-1]