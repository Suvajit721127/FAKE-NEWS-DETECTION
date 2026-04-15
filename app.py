import os
import torch
from flask import Flask, request

from model import (
    load_hinvec,
    load_trained_model,
    predict
)

# ==============================
# BASIC CONFIG
# ==============================
app = Flask(__name__)

# Reduce CPU threads (IMPORTANT for low RAM)
torch.set_num_threads(1)

# Force CPU (Render free tier safe)
device = torch.device("cpu")
print("Using device:", device)

# ==============================
# GLOBAL MODEL VARIABLES
# ==============================
model = None
tokenizer = None
hinvec_model = None
MODEL_LOADED = False


# ==============================
# LAZY LOAD MODELS (ON DEMAND)
# ==============================
def load_models():
    global model, tokenizer, hinvec_model, MODEL_LOADED

    if not MODEL_LOADED:
        try:
            print("🔄 Loading models...")

            tokenizer, hinvec_model = load_hinvec()
            model = load_trained_model()

            MODEL_LOADED = True
            print("✅ Models loaded successfully!")

        except Exception as e:
            print("❌ Model loading failed:", str(e))
            MODEL_LOADED = False


# ==============================
# HOME ROUTE
# ==============================
@app.route("/")
def home():
    return """
    <html>
    <head>
        <title>Fake News Detector</title>
        <style>
            body {
                margin: 0;
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: #fff;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }
            .box {
                background: rgba(255,255,255,0.1);
                padding: 25px;
                border-radius: 12px;
                width: 420px;
                text-align: center;
            }
            textarea {
                width: 100%;
                padding: 10px;
                border-radius: 8px;
                border: none;
                resize: none;
                margin-top: 10px;
            }
            button {
                margin-top: 15px;
                padding: 10px 20px;
                border: none;
                border-radius: 8px;
                background: #ff7eb3;
                color: white;
                cursor: pointer;
            }
        </style>
    </head>
    <body>
        <div class="box">
            <h2>📰 Fake News Detector</h2>
            <form action="/predict" method="post">
                <textarea name="news" rows="6" placeholder="Paste news here..." required></textarea>
                <br>
                <button type="submit">Analyze</button>
            </form>
        </div>
    </body>
    </html>
    """


# ==============================
# PREDICT ROUTE
# ==============================
@app.route("/predict", methods=["POST"])
def predict_news():
    load_models()

    if not MODEL_LOADED:
        return "<h2>❌ Model failed to load. Check logs.</h2>"

    text = request.form.get("news")

    if not text or text.strip() == "":
        return "<h2>❌ Please enter some text!</h2><a href='/'>Go Back</a>"

    try:
        result = predict(text, model, tokenizer, hinvec_model)

        if result == 1:
            output = "🛑 Fake News"
            color = "red"
        else:
            output = "✅ Real News"
            color = "green"

    except Exception as e:
        return f"<h2>❌ Prediction Error: {str(e)}</h2>"

    return f"""
    <html>
    <body style="text-align:center; margin-top:50px; font-family:Arial;">
        <h1 style="color:{color};">{output}</h1>
        <br>
        <a href="/">🔙 Try Again</a>
    </body>
    </html>
    """


# ==============================
# HEALTH CHECK (IMPORTANT)
# ==============================
@app.route("/health")
def health():
    return {"status": "ok"}


# ==============================
# RUN SERVER (RENDER COMPATIBLE)
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
