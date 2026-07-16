from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from app.rag.agent import start_agent

BASE_DIR = Path(__file__).resolve().parent
print(BASE_DIR)

app = FastAPI()
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.get("/", response_class=HTMLResponse)
async def search(request: Request, s: str | None = None):
    # print(q)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"title": "AI 쇼퍼"},
    )

@app.get("/query", response_class=HTMLResponse)
def query(request: Request, s: str = "대한민국 수도는?"):
    print(s)
    result = start_agent(s)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"answer": result.get("messages")[-1].content},
    )