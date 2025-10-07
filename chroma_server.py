# chromadb_server.py
from chromadb.config import Settings
from chromadb.server import FastAPIApp
import uvicorn

# ----------------------
# CONFIG
# ----------------------
# Use local SQLite storage inside Render container
CHROMA_DB_DIR = "/app/chroma_data"  # Render file system path

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
    uvicorn.run(app, host="0.0.0.0", port=8000)
