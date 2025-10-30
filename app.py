from fastapi import FastAPI
from pydantic import BaseModel
from langgraph_workflow import run_rag_pipeline

app = FastAPI()

class RagRequest(BaseModel):
    message: str
    language: str = "en"

@app.post("/rag")
async def rag_endpoint(req: RagRequest):
    try:
        answer = run_rag_pipeline(req.message, req.language)
        return {"response": answer}
    except Exception as e:
        return {"response": "none", "error": str(e)}
