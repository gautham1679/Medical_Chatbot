from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

key = os.getenv("PINECONE_API_KEY")
print(f"Loaded key: {key}")
