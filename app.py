from fastapi import FastAPI
from pydantic import BaseModel
from langgraph_workflow import run_rag_pipeline

app = FastAPI()

class RagRequest(BaseModel):
    message: str
    language: str = "en"
    role: str = "patient"  # Accepts 'doctor' or 'patient'

@app.post("/rag")
async def rag_endpoint(req: RagRequest):
    try:
        # Pass the role to the workflow
        answer = run_rag_pipeline(req.message, role=req.role)
        return {"response": answer}
    except Exception as e:
        return {"response": "none", "error": str(e)}