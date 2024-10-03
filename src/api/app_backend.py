from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
from typing import List
from src.api.controllers.controller import pipeline

app = FastAPI()

class UserPromptRequest(BaseModel):
    question: str

class AssistantResponse(BaseModel):
    answer: str
    url: List[str]


@app.post("/qa-vn-news", response_model=AssistantResponse)
def get_response(data: UserPromptRequest):
    try:
        answer, url = pipeline(data.question)
        return AssistantResponse(
            answer=answer,
            url=url
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
