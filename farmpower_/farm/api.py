from fastapi import FastAPI, Request
from farm.miner import mine_span, Ledger

app = FastAPI()
ledger = Ledger()

@app.post("/mine")
async def mine(request: Request):
    data = await request.json()
    parent_ids = data.get("parent_ids", [])
    prompt = data.get("prompt", "Genesis Content")
    span = mine_span(ledger, parent_ids, prompt)
    return {
        "id": span.id,
        "energy": span.energy,
        "signature": span.signature,
        "content": span.content
    }

@app.get("/span/{span_id}")
def get_span(span_id: str):
    span = ledger.get(span_id)
    if span:
        return {
            "id": span.id,
            "energy": span.energy,
            "content": span.content,
            "created_at": span.created_at.isoformat()
        }
    return {"error": "Span not found"}