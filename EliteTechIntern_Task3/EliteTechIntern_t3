from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import uvicorn
import os

DATASET_PATH = r"D:\shiva-py\EliteTechIntern_Task2\EliteTechIntern_Task3\spam.csv"
MODEL_PATH = Path("spam_pipeline.joblib")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

def train_and_save_model():
    print("Training model...")
    df = pd.read_csv(DATASET_PATH, encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "message"]

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", MultinomialNB())
    ])

    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, MODEL_PATH)
    print("Model trained and saved as spam_pipeline.joblib")

def load_model():
    if MODEL_PATH.exists():
        print("Model already exists. Skipping training.")
    else:
        train_and_save_model()
    return joblib.load(MODEL_PATH)

model = load_model()


if not os.path.exists("templates"):
    os.makedirs("templates")

index_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Spam Detector</title>
    <style>
        body { font-family: Arial, sans-serif; background-color: #f4f4f4; text-align: center; padding: 50px; }
        h1 { color: #333; }
        form { background: white; padding: 20px; border-radius: 10px; display: inline-block; }
        textarea { width: 300px; height: 100px; margin-bottom: 20px; }
        button { padding: 10px 20px; background-color: #28a745; border: none; color: white; cursor: pointer; }
        button:hover { background-color: #218838; }
        .result { margin-top: 20px; font-weight: bold; }
    </style>
</head>
<body>
    <h1>SMS Spam Detector</h1>
    <form method="post" action="/predict">
        <textarea name="message" placeholder="Enter your message here"></textarea><br>
        <button type="submit">Check</button>
    </form>
    {% if prediction %}
        <div class="result">Prediction: {{ prediction }}</div>
    {% endif %}
</body>
</html>
"""

with open("templates/index.html", "w", encoding="utf-8") as f:
    f.write(index_html)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, message: str = Form(...)):
    prediction = model.predict([message])[0]
    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
