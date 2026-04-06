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

# --- LIGHTWEIGHT URL SCRAPER ---
def scrape_url(url: str) -> str:
    """Fetches text from a given URL so Service B doesn't get garbage data."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = ' '.join([p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3'])])
        return text[:5000] 
    except Exception as e:
        return f"[System Note: Could not extract data from {url}]"

def yield_event(event_type: str, message: str) -> str:
    """Helper to format JSON-line streams for the frontend."""
    return json.dumps({"type": event_type, "message": message}) + "\n"

@app.post("/api/analyze")
async def agentic_orchestrator(
    prompt: Annotated[str, Form()] = "", 
    history: Annotated[str, Form()] = "[]",
    documents: Annotated[Optional[List[UploadFile]], File()] = None
):
    # 1. PRE-PROCESS FILES
    file_data = []
    if documents:
        for doc in documents:
            if doc.filename:
                content = await doc.read()
                file_data.append((doc.filename, content, doc.content_type))
    has_files = len(file_data) > 0

    # 2. PARSE CHAT HISTORY FOR MEMORY
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

    # 3. THE ASYNC GENERATOR
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

        # --- THE ROUTER ---
        yield yield_event("status", "Analyzing intent to route request...")
        
        router_prompt = f"Analyze: '{prompt}'. Files: {has_files}. URLs: {bool(urls)}. Categories: DOC_ANALYSIS, WEB_SEARCH, HYBRID_ANALYSIS, GENERAL_CHAT. Respond ONLY with category name."
        
        try:
            route_response = client.chat.completions.create(model=MODEL_ID, messages=[{"role": "user", "content": router_prompt}])
            route_check = route_response.choices[0].message.content.strip().upper()
            yield yield_event("status", f"Route Selected: {route_check}")
        except Exception as e:
            yield yield_event("error", f"Router Error: {str(e)}")
            return

        # Path: GENERAL_CHAT
        if "GENERAL_CHAT" in route_check:
            yield yield_event("status", "Synthesizing chat response...")
            # UPDATED IDENTITY HERE
            system_prompt = "You are Docify, an advanced AI document analysis engine built with TF-IDF, Cosine Similarity, and LangChain RAG. Respond politely and conversationally."
            messages = [{"role": "system", "content": system_prompt}] + formatted_contents + [{"role": "user", "content": prompt}]
            try:
                chat_response = client.chat.completions.create(model=MODEL_ID, messages=messages)
                yield yield_event("result", chat_response.choices[0].message.content)
            except Exception:
                yield yield_event("error", "Service busy. Please wait.")
            return

        # Path: WEB_SEARCH
        if "WEB_SEARCH" in route_check and "HYBRID" not in route_check:
            yield yield_event("status", "Consulting knowledge base...")
            # UPDATED IDENTITY HERE
            system_prompt = "You are Docify, an advanced AI document analysis engine built with TF-IDF, Cosine Similarity, and LangChain RAG. Use internal knowledge to answer."
            messages = [{"role": "system", "content": system_prompt}] + formatted_contents + [{"role": "user", "content": prompt}]
            try:
                search_response = client.chat.completions.create(model=MODEL_ID, messages=messages)
                yield yield_event("result", search_response.choices[0].message.content)
            except Exception:
                yield yield_event("error", "Rate limit hit.")
            return

        # --- PATH: DOC / HYBRID ANALYSIS (Calls Service B) ---
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
        rag_context_snippets = [] # 👈 To store the RAG results from Service B
        
        if not error_message and (has_files or analysis_text):
            yield yield_event("status", "Calling Data Engine (Service B)...")
            try:
                form_data = {"prompt": prompt}
                files_to_send = [("documents", (fname, fcontent, ftype)) for fname, fcontent, ftype in file_data] if has_files else []
                if analysis_text: form_data["raw_text"] = analysis_text

                engine_response = await asyncio.to_thread(
                    requests.post,
                    "http://127.0.0.1:8000/engine/analyze", # 👈 Fixed to port 8001
                    data=form_data,
                    files=files_to_send if files_to_send else None
                )
                engine_response.raise_for_status()
                result_json = engine_response.json()
                
                if result_json.get("status") == "error":
                    error_message = result_json.get("message")
                else:
                    lab_data = json.dumps(result_json.get("lab_report", []))
                    rag_context_snippets = result_json.get("rag_context", []) # 👈 Extract RAG snippets
                    
                yield yield_event("status", "Data Engine analysis complete.")
            except Exception as e:
                error_message = "Data Engine unreachable. Ensure Service B is running on port 8001."
                yield yield_event("status", "Failed to reach Data Engine.")

        # --- FINAL RAG SYNTHESIZER ---
        yield yield_event("status", "Synthesizing final report...")
        
        try:
            if error_message:
                messages = formatted_contents + [{"role": "user", "content": f"Error: {error_message}"}]
                final_response = client.chat.completions.create(model=MODEL_ID, messages=messages)
            else:
                # UPDATED FULL IDENTITY HERE
                system_msg = """You are Docify, an advanced AI document analysis engine. 
                Your core architecture utilizes TF-IDF, Cosine Similarity, and LangChain-powered RAG (Retrieval-Augmented Generation) to extract, analyze, and retrieve insights from complex documents.
                
                CRITICAL INSTRUCTIONS:
                1. Answer the user's question naturally and conversationally.
                2. ONLY IF you are analyzing a specific document or resume, structure your response to include:
                   - Applicant/Author names (if found in the text).
                   - A Recommendation based on the text and relevance scores.
                   - The exact 'word_count' provided in the JSON data.
                3. IF the user is just asking a general question about you (e.g., "What is Docify?"), proudly explain your capabilities as a document analysis tool, but DO NOT use the Applicant/Recommendation template.
                4. DO NOT explain the basic definitions of TF-IDF or Cosine Similarity unless the user explicitly asks you to. Just use the provided context to answer the prompt."""

                if "HYBRID" in route_check:
                    system_msg += "\n5. User requested web research; include your own external facts."

                # Format the retrieved RAG snippets into a readable string for the LLM
                rag_formatted = "\n\n".join([f"Relevant Snippet {i+1}: {text}" for i, text in enumerate(rag_context_snippets)])

                if not lab_data or lab_data == "[]":
                    user_content_str = f"Question: {prompt}\n\n[No new files. Use history.]"
                else:
                    user_content_str = f"""
                    User Question: {prompt}
                    
                    RETRIEVED DOCUMENT SNIPPETS (RAG):
                    {rag_formatted}
                    
                    MATHEMATICAL SCORES (TF-IDF):
                    {lab_data}
                    
                    Use the context above to answer the user's question."""

                messages = [{"role": "system", "content": system_msg}] + formatted_contents + [{"role": "user", "content": user_content_str}]
                final_response = client.chat.completions.create(model=MODEL_ID, messages=messages)

            yield yield_event("result", final_response.choices[0].message.content)
        except Exception as e:
            yield yield_event("error", f"Synthesis Error: {str(e)}")

    return StreamingResponse(process_stream(), media_type="application/x-ndjson")