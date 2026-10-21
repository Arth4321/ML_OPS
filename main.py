from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field


MODEL_PATH = Path(__file__).resolve().parent / "models" / "iris_model.joblib"
model = joblib.load(MODEL_PATH)
class_names = ["setosa", "versicolor", "virginica"]

app = FastAPI(title="Iris Random Forest API")


class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)
# above class is validation check 

@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict")
def predict(iris: IrisInput):
    features = [[
        iris.sepal_length,
        iris.sepal_width,
        iris.petal_length,
        iris.petal_width,
    ]]
    print(features)
    prediction = int(model.predict(features)[0])
    return {"Prediction_class": prediction, "class_name": class_names[prediction]}
