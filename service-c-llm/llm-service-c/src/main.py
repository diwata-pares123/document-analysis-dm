import os
import re
import json
import asyncio
import requests
from bs4 import BeautifulSoup
from typing import List, Annotated, Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from groq import Groq 

load_dotenv()
app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Initialize the Groq Client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL_ID = "llama-3.3-70b-versatile" 

def scrape_url(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = ' '.join([p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3'])])
        return text[:5000] 
    except Exception as e:
        return f"[System Note: Could not extract data from {url}]"

def yield_event(event_type: str, message: str, trigger_pdf: bool = False) -> str:
    return json.dumps({"type": event_type, "message": message, "trigger_pdf": trigger_pdf}) + "\n"

# --- FIXED 1: AI IS NO LONGER IN CHARGE OF PDFS ---
# We use a strict keyword check on the USER'S prompt. No AI hallucination possible.
def check_pdf_intent(user_prompt: str) -> bool:
    prompt_lower = user_prompt.lower()
    pdf_keywords = [
        "download pdf", "export pdf", "generate pdf", "make a pdf", 
        "download report", "export report", "create pdf"
    ]
    return any(kw in prompt_lower for kw in pdf_keywords)

@app.post("/api/analyze")
async def agentic_orchestrator(
    prompt: Annotated[str, Form()] = "", 
    history: Annotated[str, Form()] = "[]",
    documents: Annotated[Optional[List[UploadFile]], File()] = None
):
    # Lock in the PDF decision immediately based on the user's prompt
    should_trigger_pdf = check_pdf_intent(prompt)

    file_data = []
    if documents:
        for doc in documents:
            if doc.filename:
                content = await doc.read()
                file_data.append((doc.filename, content, doc.content_type))
    has_files = len(file_data) > 0

    try:
        chat_history = json.loads(history)
    except:
        chat_history = []
        
    formatted_contents = []
    for msg in chat_history:
        content_text = msg.get("content", "").strip()
        if not content_text: continue
        role = "user" if msg.get("role") == "user" else "assistant"
        formatted_contents.append({"role": role, "content": content_text})

    async def process_stream():
        yield yield_event("status", "Request received by Orchestrator...")
        await asyncio.sleep(0.1) 
        
        error_message = None 
        urls = re.findall(r'(https?://[^\s]+)', prompt)
        scraped_text = ""
        scrape_error = None
        
        if urls:
            yield yield_event("status", f"Found {len(urls)} URL(s). Extracting content...")
            for url in urls:
                scraped_result = scrape_url(url)
                if "[System Note" in scraped_result:
                    scrape_error = f"I couldn't reach or extract text from this URL: {url}"
                else:
                    scraped_text += f"\n\n--- Content from {url} ---\n{scraped_result}"

        if scrape_error and not scraped_text and not has_files:
            error_message = scrape_error

        yield yield_event("status", "Analyzing intent to route request...")
        router_prompt = f"Analyze: '{prompt}'. Files: {has_files}. URLs: {bool(urls)}. Categories: DOC_ANALYSIS, WEB_SEARCH, HYBRID_ANALYSIS, GENERAL_CHAT. Respond ONLY with category name."
        
        try:
            route_response = client.chat.completions.create(model=MODEL_ID, messages=[{"role": "user", "content": router_prompt}])
            route_check = route_response.choices[0].message.content.strip().upper()
            yield yield_event("status", f"Route Selected: {route_check}")
        except Exception as e:
            yield yield_event("error", f"Router Error: {str(e)}")
            return

        if "GENERAL_CHAT" in route_check:
            yield yield_event("status", "Synthesizing chat response...")
            system_prompt = "You are Docify, an advanced AI document analysis engine. Respond politely, organically, and conversationally."
            messages = [{"role": "system", "content": system_prompt}] + formatted_contents + [{"role": "user", "content": prompt}]
            try:
                chat_response = client.chat.completions.create(model=MODEL_ID, messages=messages)
                final_text = chat_response.choices[0].message.content
                yield yield_event("result", final_text, should_trigger_pdf)
            except Exception:
                yield yield_event("error", "Service busy. Please wait.")
            return

        if "WEB_SEARCH" in route_check and "HYBRID" not in route_check:
            yield yield_event("status", "Consulting knowledge base...")
            system_prompt = "You are Docify, an advanced AI document analysis engine. Use internal knowledge to answer naturally."
            messages = [{"role": "system", "content": system_prompt}] + formatted_contents + [{"role": "user", "content": prompt}]
            try:
                search_response = client.chat.completions.create(model=MODEL_ID, messages=messages)
                final_text = search_response.choices[0].message.content
                yield yield_event("result", final_text, should_trigger_pdf)
            except Exception:
                yield yield_event("error", "Rate limit hit.")
            return

        full_text_to_analyze = prompt + scraped_text
        analysis_text = ""
        
        if not has_files and len(full_text_to_analyze) > 100:
            yield yield_event("status", "Extracting raw data from prompt...")
            try:
                extraction_response = client.chat.completions.create(
                    model=MODEL_ID,
                    messages=[{"role": "user", "content": f"Extract ONLY raw text content from: '{full_text_to_analyze}'"}]
                )
                analysis_text = extraction_response.choices[0].message.content.strip()
            except Exception:
                analysis_text = scraped_text
        else:
            analysis_text = scraped_text

        lab_data = "[]"
        rag_context_snippets = [] 
        
        if not error_message and (has_files or analysis_text):
            yield yield_event("status", "Calling Data Engine (Service B)...")
            try:
                form_data = {"prompt": prompt}
                files_to_send = [("documents", (fname, fcontent, ftype)) for fname, fcontent, ftype in file_data] if has_files else []
                if analysis_text: form_data["raw_text"] = analysis_text

                engine_response = await asyncio.to_thread(
                    requests.post,
                    "http://127.0.0.1:8000/engine/analyze",
                    data=form_data,
                    files=files_to_send if files_to_send else None
                )
                engine_response.raise_for_status()
                result_json = engine_response.json()
                
                if result_json.get("status") == "error":
                    error_message = result_json.get("message")
                else:
                    lab_data = json.dumps(result_json.get("lab_report", []))
                    rag_context_snippets = result_json.get("rag_context", []) 
                    
                yield yield_event("status", "Data Engine analysis complete.")
            except Exception as e:
                error_message = "Data Engine unreachable. Ensure Service B is running on port 8000."
                yield yield_event("status", "Failed to reach Data Engine.")

        yield yield_event("status", "Synthesizing final report...")
        
        try:
            if error_message:
                messages = formatted_contents + [{"role": "user", "content": f"Error: {error_message}"}]
                final_response = client.chat.completions.create(model=MODEL_ID, messages=messages)
                final_text = final_response.choices[0].message.content
            else:
                # --- FIXED 2: REMOVED RIGID TEMPLATE INSTRUCTIONS ---
                system_msg = """You are Docify, an advanced AI document analysis engine. 
                Your core architecture utilizes TF-IDF, Cosine Similarity, and LangChain-powered RAG to extract and analyze insights.
                
                CRITICAL INSTRUCTIONS:
                1. Answer the user's question naturally and directly based on the provided context.
                2. DO NOT use a universal template, rigid structure, or forced formatting. Adapt your response organically to best answer the specific query without forcing sections like 'Authors' or 'Word counts' unless requested.
                3. DO NOT explain definitions of TF-IDF unless explicitly asked."""

                if "HYBRID" in route_check:
                    system_msg += "\n4. User requested web research; include your own external facts organically."

                rag_formatted = "\n\n".join([f"Relevant Snippet {i+1}: {text}" for i, text in enumerate(rag_context_snippets)])

                if not lab_data or lab_data == "[]":
                    user_content_str = f"Question: {prompt}\n\n[No new files. Use history.]"
                else:
                    user_content_str = f"User Question: {prompt}\n\nRETRIEVED DOCUMENT SNIPPETS (RAG):\n{rag_formatted}\n\nMATHEMATICAL SCORES (TF-IDF):\n{lab_data}\n\nUse the context above to answer naturally."

                messages = [{"role": "system", "content": system_msg}] + formatted_contents + [{"role": "user", "content": user_content_str}]
                final_response = client.chat.completions.create(model=MODEL_ID, messages=messages)
                final_text = final_response.choices[0].message.content

            yield yield_event("result", final_text, should_trigger_pdf)
        except Exception as e:
            yield yield_event("error", f"Synthesis Error: {str(e)}")

    return StreamingResponse(process_stream(), media_type="application/x-ndjson")