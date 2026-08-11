import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Folders
DB_PATH = "db"
DATA_PATH = "data"

# The uploaded PDF will always be saved here by app.py
PDF_PATH = os.path.join(DATA_PATH, "uploaded.pdf")