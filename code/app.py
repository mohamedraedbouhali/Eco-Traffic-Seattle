from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, create_model
import pandas as pd
import joblib
import io
import os
from typing import Optional, List

app = FastAPI(title="Weather Prediction API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "data/best_model_pipeline.pkl"
model = None
WeatherData = None
NUMERICAL_FEATURES = []
CATEGORICAL_FEATURES = []
ALL_FEATURES = []

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    try:
        # Inspect the preprocessor to find feature names
        preprocessor = model.named_steps['preprocessor']
        
        num_transformer_tuple = next(t for t in preprocessor.transformers_ if t[0] == 'num')
        cat_transformer_tuple = next(t for t in preprocessor.transformers_ if t[0] == 'cat')
        
        NUMERICAL_FEATURES = num_transformer_tuple[2]
        CATEGORICAL_FEATURES = cat_transformer_tuple[2]
        ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

        # Create a dynamic Pydantic model for input validation
        # The model pipeline handles missing values, so we can use Optional
        fields = {f: (Optional[float], None) for f in NUMERICAL_FEATURES}
        fields.update({f: (Optional[str], None) for f in CATEGORICAL_FEATURES})
        WeatherData = create_model('WeatherData', **fields)

    except (KeyError, AttributeError, StopIteration, TypeError) as e:
        print(f"Warning: Could not automatically determine features from model pipeline. Error: {e}")
        model = None # Degrade service if we can't build the input model
else:
    print(f"Warning: Model file not found at {MODEL_PATH}")

# Fallback BaseModel if model loading or feature extraction fails
if WeatherData is None:
    class WeatherData(BaseModel):
        pass

@app.get("/health")
def health_check():
    """
    Health check endpoint to ensure API is running and model is loaded.
    """
    if model:
        return {"status": "ok", "model_loaded": True}
    return {"status": "degraded", "model_loaded": False}

@app.post("/predict")
def predict_weather(data: WeatherData):
    """
    Real-time prediction for a single weather observation.
    The model predicts if the temperature is above the median (1) or not (0).
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model pipeline not available")
    
    input_df = pd.DataFrame([data.dict()])
    # Ensure columns are in the same order as during training
    input_df = input_df[ALL_FEATURES]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    return {
        "prediction": int(prediction),
        "probability_above_median": float(probability),
        "interpretation": "Above Median Temperature" if prediction == 1 else "Below or Equal to Median Temperature"
    }

@app.post("/predict_batch")
async def predict_weather_batch(file: UploadFile = File(...)):
    """
    Batch prediction on a CSV file of weather observations.
    Returns the input data with 'prediction' and 'probability_above_median' columns.
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model pipeline not available")
    if not ALL_FEATURES:
        raise HTTPException(status_code=503, detail="Model features not loaded. Cannot process batch.")
    
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # Verify columns exist
        if not all(col in df.columns for col in ALL_FEATURES):
             raise HTTPException(status_code=400, detail=f"CSV must contain all required feature columns: {ALL_FEATURES}")
             
        # Ensure correct column order and subset for prediction
        input_df = df[ALL_FEATURES]
        predictions = model.predict(input_df)
        probabilities = model.predict_proba(input_df)[:, 1]
        
        df['prediction'] = predictions
        df['probability_above_median'] = probabilities
        
        return df.to_dict(orient='records')
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)