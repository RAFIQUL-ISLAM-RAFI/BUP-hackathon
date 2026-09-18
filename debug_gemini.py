import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 50)
print("DIAGNOSTIC — Gemini Setup")
print("=" * 50)

# 1. Check .env loaded
key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL")
print(f"\n1. GEMINI_API_KEY = {key[:15] + '...' if key else '❌ NOT SET'}")
print(f"2. GEMINI_MODEL   = {model}")

if not key:
    print("\n❌ .env file-এ GEMINI_API_KEY নেই!")
    print("   Folder: E:\\problem solve\\.env")
    exit(1)

if key.startswith("your-"):
    print("\n❌ .env-এ এখনো placeholder 'your-...' আছে")
    print("   আসল key বসাও")
    exit(1)

# 2. Check library
print("\n3. Testing google.genai import...")
try:
    from google import genai
    from google.genai import types
    print("   ✅ google.genai imported OK")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    print("   Fix: pip install --upgrade google-genai")
    exit(1)

# 3. Try API call
print(f"\n4. Trying API call with model '{model or 'gemini-2.5-flash'}'...")
try:
    client = genai.Client(api_key=key)
    resp = client.models.generate_content(
        model=model or "gemini-3.6-flash",
        contents='Return JSON: {"test": "ok"}',
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
        ),
    )
    print(f"   ✅ SUCCESS!")
    print(f"   Response: {resp.text}")
except Exception as e:
    print(f"   ❌ API call failed: {e}")
    print(f"   Error type: {type(e).__name__}")

# 4. List available models
print(f"\n5. Listing available models for your API key...")
try:
    client = genai.Client(api_key=key)
    for m in client.models.list():
        print(f"   - {m.name}")
except Exception as e:
    print(f"   ❌ Could not list: {e}")