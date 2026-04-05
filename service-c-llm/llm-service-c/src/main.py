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

# 1. Swap the Google import for Groq
from groq import Groq 

load_dotenv()
app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 2. Initialize the Groq Client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# Using Llama 3.3 70B as it's an incredibly capable and fast default for Groq
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
    # 1. PRE-PROCESS FILES (Must be done before streaming starts)
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
        if not content_text:
            continue
        # Groq uses "assistant" instead of "model"
        role = "user" if msg.get("role") == "user" else "assistant"
        # Groq uses the {"role": ..., "content": ...} dictionary format
        formatted_contents.append({"role": role, "content": content_text})

    # 3. THE ASYNC GENERATOR (This broadcasts live JSON updates to the frontend)
    async def process_stream():
        yield yield_event("status", "Request received by Orchestrator...")
        await asyncio.sleep(0.2) 
        
        # Initialize error_message here to catch scrape errors early
        error_message = None 

        # --- URL EXTRACTION ---
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

        # If the scrape failed AND there are no files attached, set the global error_message 
        # so Service B gets skipped entirely.
        if scrape_error and not scraped_text and not has_files:
            error_message = scrape_error

        # --- THE ROUTER ---
        yield yield_event("status", "Analyzing intent to route request...")
        
        router_prompt = f"""
        Analyze this user input: "{prompt}"
        Has attached files: {has_files}
        Has URLs: {bool(urls)}
        
        Categorize into EXACTLY ONE of these categories:
        DOC_ANALYSIS: User wants to analyze attached files, pasted text, or provided URLs.
        WEB_SEARCH: User is asking for factual information, research links (RRLs), or real-time internet search.
        HYBRID_ANALYSIS: User wants BOTH document analysis AND web search/external links simultaneously.
        GENERAL_CHAT: User is saying hello, making small talk, or asking completely irrelevant questions.
        
        Respond ONLY with the category name.
        """
        
        try:
            # Groq completion call
            route_response = client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "user", "content": router_prompt}]
            )
            route_check = route_response.choices[0].message.content.strip().upper()
            
            yield yield_event("status", f"Route Selected: {route_check}")
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "quota" in error_str:
                yield yield_event("error", "Quota Hit: Docify is processing too many requests. Please wait a minute.")
            else:
                yield yield_event("error", f"Router Error: {str(e)}")
            return

        # -----------------------------------------
        # PATH C: GENERAL CHAT
        # -----------------------------------------
        if "GENERAL_CHAT" in route_check:
            yield yield_event("status", "Synthesizing chat response...")
            
            persona = "You are Docify. Politely respond to the user, but gently remind them your primary purpose is Cosine Similarity document analysis."
            
            # Combine System Persona + History + Latest Prompt for Groq
            messages = [{"role": "system", "content": persona}] + formatted_contents + [{"role": "user", "content": prompt}]
            
            try:
                chat_response = client.chat.completions.create(
                    model=MODEL_ID,
                    messages=messages
                )
                yield yield_event("result", chat_response.choices[0].message.content)
            except Exception:
                yield yield_event("error", "Docify is on a quick cooldown. Please wait a minute.")
            return

        # -----------------------------------------
        # PATH B: WEB SEARCH (Simulated for Groq)
        # -----------------------------------------
        if "WEB_SEARCH" in route_check and "HYBRID" not in route_check:
            yield yield_event("status", "Consulting knowledge base...")
            
            # Adjusted persona since Groq cannot actively browse the internet
            persona = "You are Docify, a helpful research assistant. Provide clear, factual answers based on your training data."
            
            messages = [{"role": "system", "content": persona}] + formatted_contents + [{"role": "user", "content": prompt}]
            
            try:
                search_response = client.chat.completions.create(
                    model=MODEL_ID,
                    messages=messages
                )
                yield yield_event("result", search_response.choices[0].message.content)
            except Exception:
                yield yield_event("error", "Rate limit hit. Be right back in a minute.")
            return

        # -----------------------------------------
        # PATH A & D: DOC ANALYSIS OR HYBRID ANALYSIS
        # (Call Service B)
        # -----------------------------------------
        full_text_to_analyze = prompt + scraped_text
        analysis_text = ""
        
        if not has_files and len(full_text_to_analyze) > 100:
            yield yield_event("status", "Extracting raw data from prompt...")
            try:
                split_query = f"Extract ONLY the raw data/text content. Exclude user commands. Input: '{full_text_to_analyze}'"
                # Groq extraction call
                extraction_response = client.chat.completions.create(
                    model=MODEL_ID,
                    messages=[{"role": "user", "content": split_query}]
                )
                analysis_text = extraction_response.choices[0].message.content.strip()
            except Exception:
                analysis_text = scraped_text
        else:
            analysis_text = scraped_text

        lab_data = "[]"
        
        # Only call Data Engine if no scrape errors occurred
        if not error_message and (has_files or analysis_text):
            yield yield_event("status", "Calling Data Engine (Service B)...")
            
            try:
                form_data = {"prompt": prompt}
                files_to_send = []
                
                if has_files:
                    files_to_send = [("documents", (fname, fcontent, ftype)) for fname, fcontent, ftype in file_data]
                if analysis_text:
                    form_data["raw_text"] = analysis_text

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
                    
                yield yield_event("status", "Data Engine analysis complete.")
                
            except Exception as e:
                error_message = "The Data Engine was unreachable. Ensure Service B is running."
                yield yield_event("status", "Failed to reach Data Engine.")

        # --- FINAL SYNTHESIZER ---
        if "HYBRID" in route_check:
            yield yield_event("status", "Synthesizing hybrid report...")
        else:
            yield yield_event("status", "Synthesizing final report...")
        
        try:
            if error_message:
                messages = formatted_contents + [{"role": "user", "content": f"Tell the user gracefully that we encountered an error reading the document: {error_message}"}]
                final_response = client.chat.completions.create(model=MODEL_ID, messages=messages)
            else:
                system_msg = """You are the Docify Assistant. Translate the raw TF-IDF and Cosine Similarity scores into simple, conversational insights. 
                DO NOT use a rigid template or list all the use cases. Instead, dynamically pick just 1 or 2 real-world NLP use cases (like Plagiarism Detection, Search Ranking, or Semantic Search) that actually make sense for the user's specific document.
                Keep it flowing naturally. You can briefly mention that this evaluates 'semantic vector orientation ignoring document length', but weave it in smoothly so you don't sound like a textbook.
                
                CRITICAL INSTRUCTIONS FOR RESUME/DOCUMENT ANALYSIS:
                1. You will receive 'full_extracted_text' in the Data Engine Output.
                2. You MUST explicitly state the names of the applicants or authors found in this text.
                3. If you do not see a name in the text, refer to them by their filename.
                4. Do not invent skills or facts that are not explicitly written in the extracted text.
                5. Base your final recommendation heavily on both the Data Engine's Relevance Score AND the actual information found in the text."""
                
                if "HYBRID" in route_check:
                    system_msg += "\n\nCRITICAL: The user also requested external links/research. Use your internal knowledge to provide relevant facts and context alongside the document analysis."

                # 👇 THIS IS THE FIX WE ADDED: Handling empty lab_data so the AI uses chat history
                if lab_data == "[]" or not lab_data:
                    user_content_str = f"User Question: {prompt}\n\n[System Note: No new files were uploaded in this turn. Do not say the Data Engine failed. Please refer to the chat history above to recall the applicant names, skills, and Cosine Similarity scores to answer the user's question.]"
                else:
                    user_content_str = f"User Question: {prompt}\n\nData Engine Output: {lab_data}\n\nPlease explain these Cosine Similarity results to the user based on the context above."
                
                messages = [{"role": "system", "content": system_msg}] + formatted_contents + [{"role": "user", "content": user_content_str}]
                
                final_response = client.chat.completions.create(
                    model=MODEL_ID,
                    messages=messages
                )

            # Sends the final text directly to the chat bubble payload
            yield yield_event("result", final_response.choices[0].message.content)

        except Exception as e:
            yield yield_event("error", f"Synthesis Error: Could not generate final text. {str(e)}")

    # 3. RETURN THE STREAM
    # Now serving proper line-delimited JSON chunks!
    return StreamingResponse(process_stream(), media_type="application/x-ndjson")