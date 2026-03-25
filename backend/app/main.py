from fastapi import FastAPI, UploadFile, File, HTTPException

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
    
#JSON Response
    return {
        "total_fish": 1,
        "detections": [
            {
                "species": "clownfish",
                "confidence": 0.94
            }
        ],
        "species_counts": {
            "clownfish": 1
        }
    }
