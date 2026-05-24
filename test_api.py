from dotenv import load_dotenv
from google import genai
import os

load_dotenv()

client = genai.Client(
    api_key=os.environ.get("GOOGLE_API_KEY")
)

try:
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents="Hello"
    )

    print(response.text)

except Exception as e:
    print(type(e))
    print(e)