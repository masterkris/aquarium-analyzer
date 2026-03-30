import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from src.detection.detect import detect_fish

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Aquarium Analyzer API running"}

# add routes and schemas later as we progress

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
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
    
    #JSON file but with info from detection
    return {
        "total_fish": total_fish,
        "detections": formatted_detections,
        "species_counts": species_counts
    }