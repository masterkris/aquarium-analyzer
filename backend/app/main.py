import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from src.detection.detect import detect_fish

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
API_KEY = os.getenv("API_KEY")
gemini_client = genai.Client(api_key=API_KEY)

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"

app = FastAPI()


@app.get("/")
def read_root():
    return FileResponse(FRONTEND_DIR / "index.html", media_type="text/html")


@app.get("/favicon.svg")
def serve_favicon():
    return FileResponse(FRONTEND_DIR / "favicon.svg", media_type="image/svg+xml")


# add routes and schemas later as we progress
@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...), tankSize: int = Form(...)):
    #checking if file is supported
    allowed_types = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Ivalid file type"
        )
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="File is empty.")
    except Exception:
        raise HTTPException(status_code=500, detail="Error reading file.")
    
    temp_image_path = None #creating temp file for detect.py to read
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(image_bytes)
            temp_image_path = temp_file.name

        #calls ML pipeline
        raw_detections = detect_fish(image_path=temp_image_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
    finally:
        #delete the temp file
        if temp_image_path and os.path.exists(temp_image_path): 
            os.remove(temp_image_path)
            
    #formatted output for the frontedn   
    total_fish = len(raw_detections)
    formatted_detections = []
    species_counts = {}

    for det in raw_detections:
        #getting data from dictionary returned from detect.py
        species = det["class_name"]
        confidence = round(det["conf"], 2)

        formatted_detections.append({
            "species": species,
            "confidence": confidence
        })

        species_counts[species] = species_counts.get(species, 0) + 1

    #get stocking recommendations from Gemini
    prompt = (
        f"I have a {tankSize}-gallon freshwater aquarium with the following fish: "
        f"{', '.join(f'{count} {species}' for species, count in species_counts.items())}. "
        f"Total fish: {total_fish}. "
        "Is this tank overstocked, understocked, or appropriately stocked? "
        "Are there any compatibility issues between these species? "
        "Provide brief, actionable recommendations."
    )

    try:
        gemini_response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
        )
        recommendations = gemini_response.text
    except Exception as e:
        print(f"Gemini API error: {e}")
        recommendations = "Unable to generate recommendations at this time."

    return {
        "total_fish": total_fish,
        "detections": formatted_detections,
        "species_counts": species_counts,
        "recommendations": recommendations,
    }

app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")