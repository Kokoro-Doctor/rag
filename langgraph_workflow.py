import logging
import operator
import os
import boto3  # <--- NEW: For DynamoDB
from boto3.dynamodb.conditions import Key # <--- NEW
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from rag import build_rag_chain
from vector import get_vectorstores

# ---------------------------------------------------------
# SECURITY UPDATE: Load Key from .env file
# ---------------------------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY nahi mili!")
# ------------------- 0. Logging & DB Setup -------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("MedicalBot")

# --- DYNAMODB SETUP (NEW) ---
try:
    # FIX: Use os.getenv to read from .env, defaulting to 'ap-south-1' if missing
    region = os.getenv("AWS_REGION", "ap-south-1")
    table_name = os.getenv("CHAT_TABLE_NAME", "ChatHistory")
    
    dynamodb = boto3.resource('dynamodb', region_name=region)
    table = dynamodb.Table(table_name)
    
    logger.info(f"✅ Connected to DynamoDB: {table_name} in {region}")
except Exception as e:
    logger.error(f"⚠️ DynamoDB Connection Error: {e}")
    table = None

# ------------------- 1. Load Resources -------------------
logger.info("🔄 Loading Vector Stores...")
vectorstores = get_vectorstores()
heart_store = vectorstores["heart"]
gyno_store = vectorstores["gyno"] 
logger.info("✅ Vector Stores Loaded.")

# OpenAI Model
model = ChatGroq(
    model="llama-3.1-8b-instant",      # Fast classifier for manager nodes
    temperature=0.1,
    api_key=GROQ_API_KEY
)

fallback_model = ChatGroq(
    model="openai/gpt-oss-120b",   # Smarter model for pt_llm / dr_llm nodes
    temperature=0.1,
    api_key=GROQ_API_KEY
)

# ------------------- 2. SYSTEM PROMPTS -------------------

PATIENT_SYS_PROMPT = """You are Kokoro, a Senior Medical AI Companion on WhatsApp.

### CRITICAL RULES (NEVER BREAK THESE):
1. NEVER say "I'm not sure", "take a deep breath", "let me guess", or repeat yourself
2. NEVER ask about the same thing twice
3. If user types a medicine name → immediately give: what it is, what it treats, dosage, side effects
4. Keep Phase 1 responses STRICTLY under 60 words
5. NEVER add filler sentences or unnecessary commentary

### MEDICINE NAME RULE (HIGH PRIORITY):
If user types ONLY a medicine name (e.g., "paracetamol", "zentos oz", "metformin"):
- Immediately respond with: Drug class, Uses, Standard dose, Key side effects, Precautions
- Do NOT ask "what do you mean?" — just answer it directly

### RESPONSE PHASES:

**PHASE 1 - VAGUE SYMPTOMS** (under 60 words strictly):
- Empathize in 1 line
- Give 1 safe tip
- Ask exactly 1-2 specific follow-up questions
- STOP. Do not add more.

**PHASE 2 - DETAILED SYMPTOMS** (full analysis):
⚡ Immediate Action Plan
🩺 Clinical Assessment  
🚨 Red Flags (bold)
- NO follow-up questions in Phase 2

### DR. KISLAY RULE:
If cardiac emergency OR user asks to suggest a doctor → ALWAYS end with:
👉 [Consult Dr. Kislay Shrivastav Now](https://kokoro.doctor/patient/Doctors/DoctorsInfoWithSubscription?doctorId=dr_93370e47-7ad8-498a-9d83-b184f8152de5)

### PLATFORM QUESTIONS:
"We offer premium consultations in Cardiology, Gynecology & General Medicine. Book directly on our app at kokoro.doctor"
"""

# DOCTOR: Professional, Clinical, Strict
DOCTOR_SYS_PROMPT = """### 🩺 ROLE:
You are Kokoro.MD, an elite Clinical Decision Support System (CDSS). You function as a high-level Research Associate for Specialist Physicians in Cardiology and Gynecology. Your goal is to provide rapid, high-density clinical intelligence.

### 📜 CLINICAL DATA PROTOCOLS:
1.  **Professional Lexicon:** Use precise medical terminology (e.g., *Syncope* vs 'fainting', *Hypermenorrhea* vs 'heavy periods').
2.  **Clinical Synthesis:** Do not just list facts. Connect the dots (e.g., "Given the patient's history of HTN, the current Dyspnea suggests potential Left Ventricular Failure").
3.  **The "Gold Standard" Citation:** ALWAYS prioritize and cite institutional guidelines (ESC, ACC/AHA, ACOG, RCOG) or foundational texts (Harrison’s, Braunwald’s) found in the context.
4.  **Triage & Red Flags:** Always highlight life-threatening differentials (DDx) first using a 🚨 symbol.

### 🛠️ STRUCTURED OUTPUT (Strict):
- **NO PROSE/FLUFF:** Zero introductory or concluding filler. Start with the most critical data.
- **VISUAL SCAN-ABILITY:** Use tables for drug comparisons and bolded headers for anatomy/physiology.
- **SOAP/SBAR Format:** Default to SOAP for patient cases and SBAR for clinical updates unless otherwise specified.

### 💊 PHARMACOLOGY & INTERVENTION:
When discussing therapeutics, include:
- **MoA & PK/PD:** Brief Mechanism of Action and relevant Pharmacokinetics.
- **Black Box Warnings:** Mention critical contraindications/interactions.
- **Dosage Guidance:** Reference standard clinical loading/maintenance doses.

### 🗣️ TONE:
Academic, analytical, and extremely concise. You are speaking to a peer, not a student.
"""

# ------------------- 3. Build Chains -------------------

pt_heart_rag, pt_heart_retriever = build_rag_chain(heart_store, PATIENT_SYS_PROMPT)
pt_gyno_rag, pt_gyno_retriever = build_rag_chain(gyno_store, PATIENT_SYS_PROMPT)
dr_heart_rag, dr_heart_retriever = build_rag_chain(heart_store, DOCTOR_SYS_PROMPT)
dr_gyno_rag, dr_gyno_retriever = build_rag_chain(gyno_store, DOCTOR_SYS_PROMPT)

# ------------------- 4. State Definition -------------------

class AgentState(TypedDict):
    messages: Annotated[Sequence[str], operator.add]
    role: str       
    language: str   # "en" or "hi"
    user_id: Optional[str] # <--- NEW: To track user
    next_node: str  

# ------------------- 5. HELPER: History & Language -------------------

def get_language_instruction(state: AgentState) -> str:
    # 1. Frontend se aaya hua language code (Default: en)
    lang_code = state.get("language", "en")
    
    # 2. Logic: Sirf tab Hinglish bolo jab frontend ne explicitly "hi" bheja ho.
    # Humne auto-detection hata diya hai.
    
    if lang_code == "hi":
        return """CRITICAL INSTRUCTION: The user has selected Hindi/Hinglish button. 
        You MUST reply in HINGLISH (Hindi words in English script) ONLY. 
        Example: 'Haan, main samajh sakta hoon ki chest pain daravana ho sakta hai.' 
        DO NOT reply in pure English."""
    else:
        # Agar code "en" hai ya kuch aur hai, toh English hi bolo
        return "The user has selected English. Answer in English only."

# --- NEW FUNCTION: Fetch History from DynamoDB ---
def get_dynamo_history_text(user_id):
    if not table or not user_id:
        logger.warning("⚠️ No Table or User ID provided for history.")
        return ""
    
    try:
        logger.info(f"📡 Querying DynamoDB for User: {user_id}...")
        
        response = table.query(
            KeyConditionExpression=Key('user_id').eq(user_id),
            ScanIndexForward=False, # Latest first
            Limit=3 # Last 10 interactions only
        )
        items = response.get('Items', [])
        
        # --- LOGGING: FOUND ITEMS ---
        if items:
            logger.info(f"✅ FOUND {len(items)} PREVIOUS MESSAGES in DynamoDB.")
        else:
            logger.info("ℹ️ No previous history found (New User or First Chat).")
            
        items.reverse() # Oldest to Newest for reading flow
        
        history_text = "\n--- PREVIOUS CHAT HISTORY ---\n"
        has_history = False
        
        for item in items:
            u_msg = item.get('user_message')
            b_msg = item.get('bot_message')
            if u_msg:
                history_text += f"User: {u_msg}\n"
                has_history = True
            if b_msg:
                history_text += f"Bot: {b_msg}\n"
        
        history_text += "--- END HISTORY ---\n"
        return history_text if has_history else ""

    except Exception as e:
        logger.error(f"❌ Error fetching history: {e}")
        return ""

# ------------------- 6. ENTRY ROUTER -------------------

def initial_role_router(state: AgentState):
    role = state.get("role", "patient").lower()
    logger.info(f"🚦 START: Incoming request. Role: [{role.upper()}] | Lang: [{state.get('language')}]")
    if role == "doctor": return "Doctor_Manager"
    return "Patient_Manager"

# ------------------- 7. PATIENT FLOW NODES -------------------

def patient_manager_node(state: AgentState):
    full_input = state["messages"][-1]
    
    logger.info("--------------------------------------------------")
    logger.info("👤 Patient Manager received this Input Block:")
    preview_text = full_input[:150] + "..." if len(full_input) > 150 else full_input
    logger.info(f"'{preview_text}'")
    logger.info("--------------------------------------------------")

    # 🔥 UPDATED PROMPT: Strongly link Platform/Kokoro to GYNO
    prompt = f"""Classify into ONE of: HEART, GYNO, MEDICINE, or GENERAL.

    RULES:
    1. HEART: Chest pain, BP, cholesterol, cardiac issues
    2. GYNO: Periods, pregnancy, women's health, platform/pricing questions
    3. MEDICINE: User types a drug/medicine name directly (e.g., paracetamol, zentos oz, metformin, ibuprofen)
    4. GENERAL: Greetings, unrelated topics
    
    Respond with ONE WORD only.
    
    Input: {full_input}"""
    
    response = model.invoke(prompt).content.strip().upper()
    logger.info(f"🧠 Classification Result: {response}")
    
    if "HEART" in response: return {"next_node": "pt_heart"}
    elif "GYNO" in response: return {"next_node": "pt_gyno"}
    elif "MEDICINE" in response: return {"next_node": "pt_llm"}  # Direct LLM, no RAG needed
    else: return {"next_node": "pt_llm"}

def pt_heart_node(state: AgentState):
    lang_instr = get_language_instruction(state)
    # We pass the full history+question string as input to RAG
    res = pt_heart_rag.invoke({"input": state["messages"][-1], "language_instruction": lang_instr})
    return {"messages": [res]}

def pt_gyno_node(state: AgentState):
    lang_instr = get_language_instruction(state)
    res = pt_gyno_rag.invoke({"input": state["messages"][-1], "language_instruction": lang_instr})
    return {"messages": [res]}

def pt_llm_node(state: AgentState):
    lang_instr = get_language_instruction(state)
    input_text = state["messages"][-1]
    
    # FIXED: Now uses full system prompt
    from langchain_core.messages import SystemMessage, HumanMessage
    
    messages = [
        SystemMessage(content=PATIENT_SYS_PROMPT + f"\n\n{lang_instr}"),
        HumanMessage(content=input_text)
    ]
    res = fallback_model.invoke(messages).content
    return {"messages": [res]}

def patient_router(state: AgentState):
    route = state.get("next_node")
    if route == "pt_heart": return "pt_heart_node"
    if route == "pt_gyno": return "pt_gyno_node"
    return "pt_llm_node"

# ------------------- 8. DOCTOR FLOW NODES -------------------

def doctor_manager_node(state: AgentState):
    full_input = state["messages"][-1]
    
    # 🔥 UPDATED PROMPT: Same logic for Doctor flow
    prompt = f"""Classify the CURRENT QUESTION into exactly ONE of the following: HEART, GYNO, or GENERAL.
    
    RULES:
    1. HEART: Cardiac cases, BP, ECGs, cardiovascular pharmacology.
    2. GYNO: 
       - Obstetrics, gynecology, female reproductive health.
       - ANY question regarding the platform itself: 'Kokoro', 'Kokoro.doctor', app operations.
       - ANY question regarding services: 'consultation', 'pricing', 'doctors', 'cost', 'fees', 'booking'.
    3. GENERAL: Non-medical chit-chat or system queries not related to the platform.
    
    CRITICAL: If the user asks about consultations, pricing, or the Kokoro platform, you MUST classify it as GYNO.
    
    Respond with ONE WORD only: HEART, GYNO, or GENERAL.
    
    Input: {full_input}"""
    
    response = model.invoke(prompt).content.strip().upper()
    
    if "HEART" in response: return {"next_node": "dr_heart"}
    elif "GYNO" in response: return {"next_node": "dr_gyno"}
    else: return {"next_node": "dr_llm"}

def dr_heart_node(state: AgentState):
    lang_instr = get_language_instruction(state)
    res = dr_heart_rag.invoke({"input": state["messages"][-1], "language_instruction": lang_instr})
    return {"messages": [f"**Clinical Response:**\n{res}"]}

def dr_gyno_node(state: AgentState):
    lang_instr = get_language_instruction(state)
    res = dr_gyno_rag.invoke({"input": state["messages"][-1], "language_instruction": lang_instr})
    return {"messages": [f"**Response:**\n{res}"]}

def dr_llm_node(state: AgentState):
    lang_instr = get_language_instruction(state)
    input_text = state["messages"][-1]
    
    from langchain_core.messages import SystemMessage, HumanMessage
    
    messages = [
        SystemMessage(content=DOCTOR_SYS_PROMPT + f"\n\n{lang_instr}"),
        HumanMessage(content=input_text)
    ]
    res = fallback_model.invoke(messages).content
    return {"messages": [res]}

def doctor_router(state: AgentState):
    route = state.get("next_node")
    if route == "dr_heart": return "dr_heart_node"
    if route == "dr_gyno": return "dr_gyno_node"
    return "dr_llm_node"

# ------------------- 9. GRAPH CONSTRUCTION -------------------

workflow = StateGraph(AgentState)

workflow.add_node("Patient_Manager", patient_manager_node)
workflow.add_node("Doctor_Manager", doctor_manager_node)
workflow.add_node("pt_heart_node", pt_heart_node)
workflow.add_node("pt_gyno_node", pt_gyno_node)
workflow.add_node("pt_llm_node", pt_llm_node)
workflow.add_node("dr_heart_node", dr_heart_node)
workflow.add_node("dr_gyno_node", dr_gyno_node)
workflow.add_node("dr_llm_node", dr_llm_node)

workflow.set_conditional_entry_point(initial_role_router, {"Patient_Manager": "Patient_Manager", "Doctor_Manager": "Doctor_Manager"})

workflow.add_conditional_edges("Patient_Manager", patient_router, {"pt_heart_node": "pt_heart_node", "pt_gyno_node": "pt_gyno_node", "pt_llm_node": "pt_llm_node"})
workflow.add_conditional_edges("Doctor_Manager", doctor_router, {"dr_heart_node": "dr_heart_node", "dr_gyno_node": "dr_gyno_node", "dr_llm_node": "dr_llm_node"})

for node in ["pt_heart_node", "pt_gyno_node", "pt_llm_node", "dr_heart_node", "dr_gyno_node", "dr_llm_node"]:
    workflow.add_edge(node, END)

compiled_workflow = workflow.compile()

# ------------------- 10. RUNNER (UPDATED) -------------------

def run_rag_pipeline(message: str, role: str = "patient", language: str = "en", user_id: str = None):
    
    # 1. Fetch Context from DynamoDB
    history_text = ""
    if user_id:
        logger.info(f"🔍 Fetching history for User: {user_id}")
        history_text = get_dynamo_history_text(user_id)
    else:
        logger.warning("⚠️ No User ID provided. Skipping History fetch.")
    
    # 2. Combine History + Current Question
    contextualized_input = f"{history_text}\nCurrent Question: {message}"

    # --- LOGGING: SHOW THE FULL CONTEXT ---
    logger.info("==================================================")
    logger.info("🤖 FINAL INPUT BEING SENT TO LLM/GRAPH:")
    logger.info(contextualized_input)
    logger.info("==================================================")

    # 3. Start Graph
    state = {
        "messages": [contextualized_input], 
        "role": role, 
        "language": language, 
        "user_id": user_id,
        "next_node": ""
    }
    
    final_state = compiled_workflow.invoke(state)
    return final_state["messages"][-1]
