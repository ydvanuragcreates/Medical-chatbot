import os
import google.generativeai as genai
from dotenv import load_dotenv

print("--- Starting Test ---")

# Load the .env file
load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    print("❌ ERROR: GOOGLE_API_KEY not found in .env file.")
else:
    print("✅ API Key found in environment.")
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content("Please confirm you can see this message.")

        print("\n--- SUCCESS ---")
        print("Gemini response:")
        print(response.text)
        print("-----------------")

    except Exception as e:
        print(f"\n--- FAILED ---")
        print(f"An error occurred while contacting Google API: {e}")
        print("-----------------")
        