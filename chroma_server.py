# chromadb_server.py
import os
from chromadb.server.fastapi import FastAPIApp
from chromadb.config import Settings
import uvicorn

# ----------------------
# CONFIG
# ----------------------
CHROMA_DB_DIR = "/app/chroma_data"  # Directory inside Render container

settings = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=CHROMA_DB_DIR,
    anonymized_telemetry=False
)

app = FastAPIApp(settings)

# ----------------------
# RUN SERVER
# ----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
