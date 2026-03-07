from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Aquarium Analyzer API running"}

# add routes and schemas later as we progress