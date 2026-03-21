from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

FRONTEND_DIR = FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"

app = FastAPI()

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def read_root():
    return FileResponse(f"{FRONTEND_DIR}/index.html")

# add routes and schemas later as we progress