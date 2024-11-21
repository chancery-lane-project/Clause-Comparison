from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModel
import utils
import numpy as np
import os
from fastapi import HTTPException

app = FastAPI()

# Enable CORS for your frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and embeddings
# NOTE: This still requires the user to have some things stored locally including the utils file
local_model_dir = "../legalbert/legalbert_model"
embeddings_dir = "../legalbert/legalbert_embeddings"
tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
model = AutoModel.from_pretrained(local_model_dir)
documents, file_names = utils.load_clauses("../data/cleaned_clauses")

embeddings = {}
for method in ["cls", "mean", "max", "concat", "specific"]:
    embedding_path = os.path.join(embeddings_dir, f"{method}_embeddings.npy")
    if os.path.exists(embedding_path):
        embeddings[method] = np.load(embedding_path)
    else:
        embeddings[method] = np.vstack(
            [utils.encode_text(doc, tokenizer, model, method) for doc in documents]
        )


@app.post("/find_clauses/")
async def find_clauses(file: UploadFile, method: str = Form("cls")):
    content = await file.read()
    query = content.decode("utf-8")
    query_embedding = utils.encode_text(query, tokenizer, model, method).reshape(1, -1)
    document_embeddings = embeddings[method]
    _, _, _, similarities, _ = utils.get_matching_clause(
        query_embedding, document_embeddings, file_names
    )
    best_match_names, best_match_scores, _ = utils.find_top_three(
        similarities, file_names
    )
    return {
        "matches": [
            {"name": name.replace(".txt", ""), "score": float(score)}
            for name, score in zip(best_match_names, best_match_scores)
        ]
    }


# allow the user to see the clause
@app.get("/clause/{clause_name}")
def get_clause(clause_name: str):
    try:
        clause_filename = f"{clause_name}.txt"
        if clause_filename in file_names:
            index = file_names.index(clause_filename)
            clause_text = documents[index]
            return {"name": clause_name, "text": clause_text}
        else:
            raise HTTPException(status_code=404, detail="Clause not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return HTMLResponse(
        """
        <!DOCTYPE html>
        <html>
        <body>
        <h2>Welcome to TCLP Clause Matcher</h2>
        <p>Use a frontend to interact with this backend.</p>
        </body>
        </html>
        """
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
