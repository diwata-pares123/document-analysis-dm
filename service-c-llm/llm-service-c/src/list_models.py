import os
import google.generativeai as genai
from dotenv import load_dotenv

# 1. Load your API key from the .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# 2. Configure the SDK
genai.configure(api_key=api_key)

print("🔍 Checking available models for your API key...\n")

try:
    # 3. Ask Google for the list and print the ones that can read text/documents
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
    print("\n✅ Done! Pick one of the names above.")
except Exception as e:
    print(f"❌ Error checking models: {e}")