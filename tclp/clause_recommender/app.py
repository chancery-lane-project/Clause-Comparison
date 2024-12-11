from fastapi import FastAPI, UploadFile, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from transformers import AutoTokenizer, AutoModel
from tclp.clause_recommender import utils
import numpy as np
import os
from fastapi import HTTPException
from fastapi.responses import FileResponse
from typing import List

app = FastAPI()

# Enable CORS for your frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic authentication
security = HTTPBasic()

# Dummy credentials for demonstration purposes
USERNAME = "father"
PASSWORD = "christmas"

# Verify credentials
def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = credentials.username == USERNAME
    correct_password = credentials.password == PASSWORD
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )

# Load model and embeddings
# NOTE: This still requires the user to have some things stored locally including the utils file
local_model_dir = "/app/tclp/legalbert/legalbert_model"
embeddings_dir = "/app/tclp/legalbert/legalbert_embeddings"
tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
model = AutoModel.from_pretrained(local_model_dir)
documents, file_names = utils.load_clauses("/app/tclp/data/clause_boxes")

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
async def find_clauses(file: UploadFile):
    content = await file.read()
    method = "mean"
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
def read_root(credentials: HTTPBasicCredentials = Depends(verify_credentials)):
    return FileResponse("/app/tclp/clause_recommender/index.html")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)