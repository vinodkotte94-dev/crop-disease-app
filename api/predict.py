from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastai.vision.all import load_learner, PILImage
from pathlib import Path

app = FastAPI()

# Load your trained model (smaller model preferred for Vercel)
MODEL_PATH = Path("model.pkl")
model = load_learner(MODEL_PATH)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = PILImage.create(file.file)
    pred_class, _, _ = model.predict(img)
    return JSONResponse({"prediction": str(pred_class)})
