from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Annotated, Optional
import uvicorn

# --- UPDATED IMPORT ---
from .engine import run_intelligent_analysis, run_rag_retrieval
from .extractors import extract_text_from_file

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/engine/analyze")
async def calculate_ml_scores(
    prompt: Annotated[str, Form()] = "",
    documents: Annotated[Optional[List[UploadFile]], File()] = None,
    raw_text: Annotated[Optional[str], Form()] = None
):
    try:
        texts = []
        filenames = []

        # 1. FILE EXTRACTION & VALIDATION
        if documents and any(doc.filename for doc in documents):
            for doc in documents: 
                content = await doc.read()
                try:
                    text = extract_text_from_file(content, doc.filename)
                    if not text or not text.strip():
                        return {"status": "error", "message": f"The file '{doc.filename}' contains no readable text."}
                    texts.append(text)
                    filenames.append(doc.filename)
                except Exception as e:
                    return {"status": "error", "message": f"Extraction failed on '{doc.filename}': {str(e)}"}
                    
        elif raw_text:
            texts.append(raw_text)
            filenames.append("Chatbox_Text_Input")

        if not texts:
            return {"status": "error", "message": "No valid text found to analyze."}

        # 2. RUN ANALYSES
        # Original TF-IDF Math
        analysis_results = run_intelligent_analysis(prompt, texts)
        
        # --- NEW: RUN RAG RETRIEVAL ---
        # This finds the specific 3 most relevant paragraphs across all documents
        rag_snippets = run_rag_retrieval(prompt, texts)

        # 3. FORMAT THE DATA FOR SERVICE C
        lab_report = []
        for i in range(len(filenames)):
            exact_word_count = len(texts[i].split())
            
            lab_report.append({
                "filename": filenames[i],
                "word_count": exact_word_count,
                "tfidf_score": analysis_results[i]["score"],
                "top_keywords": analysis_results[i]["keywords"],
                "full_extracted_text": texts[i] 
            })
        
        # We return the lab_report AND the specialized RAG context
        return {
            "status": "success", 
            "lab_report": lab_report,
            "rag_context": rag_snippets  # 👈 The "Retrieved" part of RAG
        }

    except Exception as e:
        print(f"ML Engine Error: {e}")
        raise HTTPException(status_code=500, detail="The ML Engine failed to compute scores.")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)