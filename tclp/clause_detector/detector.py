from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import detector_utils as du
import os
import shutil

app = FastAPI()
MAX_FILE_LIMIT = 1000

# frontend and backend to communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load model when application starts
model_name = "clause_identifier_model.pkl"
model = du.load_model(model_name)


@app.post("/process/")
async def process_contract(files: list[UploadFile], is_folder: str = Form("false")):
    """
    Endpoint to process a contract file or folder.
    """
    if len(files) > MAX_FILE_LIMIT:
        return JSONResponse(
            content={
                "error": f"This server can only handle up to {MAX_FILE_LIMIT} files; please try again."
            },
            status_code=400,  # Bad Request
        )

    try:
        # temp directory to store the uploaded files
        temp_dir = "temp_uploads"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        if is_folder == "true":
            print("Processing folder upload...")
            file_paths = []

            for file in files:
                if not file.filename.endswith(".txt"):
                    print(f"Skipping non-txt file: {file.filename}")
                    continue

                file_path = os.path.join(temp_dir, file.filename)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "wb") as f:
                    f.write(await file.read())
                file_paths.append(file_path)
                print(f"Stored file: {file_path}")

            if not file_paths:
                return JSONResponse(
                    content={
                        "error": "No valid .txt files found in the uploaded folder."
                    },
                    status_code=400,
                )

            processed_contracts = du.load_unlabelled_contract(temp_dir)

        else:
            print("Processing single file upload...")

            # handle single file upload
            file = files[0]
            if not file.filename.endswith(".txt"):
                return JSONResponse(
                    content={
                        "error": "Only .txt files are supported for single file uploads."
                    },
                    status_code=400,
                )

            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())

            processed_contracts = du.load_unlabelled_contract(file_path)

        results = model.predict(processed_contracts["text"])

        contract_df = du.create_contract_df(
            processed_contracts["text"], processed_contracts, results, labelled=False
        )

        likely, very_likely, extremely_likely, none = du.create_threshold_buckets(
            contract_df
        )

        if is_folder == "true":
            response = du.print_percentages(
                likely,
                very_likely,
                extremely_likely,
                none,
                contract_df,
                return_result=True,
            )
        else:
            result = du.print_single(
                likely, very_likely, extremely_likely, none, return_result=True
            )
            response = {"classification": result}

        print(response)

        # cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        return JSONResponse(content=response)

    except Exception as e:
        print(f"Error processing contract: {e}")
        return JSONResponse(
            content={"error": f"An error occurred: {str(e)}"}, status_code=500
        )


@app.get("/")
def read_root():
    return HTMLResponse(
        """
        <!DOCTYPE html>
        <html>
        <body>
        <h2>Welcome to TCLP Clause Detector</h2>
        <p>Use a frontend to interact with this backend.</p>
        </body>
        </html>
        """
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("detector:app", host="127.0.0.1", port=8080, reload=True)
