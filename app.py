from flask import Flask, request
import torch

from model import (
    load_hinvec,
    load_trained_model,
    predict
)

app = Flask(__name__)

# ==============================
# DEVICE
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==============================
# LOAD MODELS (SAFE)
# ==============================
try:
    print("Loading Hinvec model...")
    tokenizer, hinvec_model = load_hinvec()

    print("Loading trained classifier...")
    model = load_trained_model()

    print("✅ All models loaded successfully!")

except Exception as e:
    print("❌ ERROR while loading models:", e)
    model = None
    tokenizer = None
    hinvec_model = None

# ==============================
# HOME PAGE
# ==============================
@app.route("/")
def home():
    return """
    <html>
    <head>
        <title>Fake News Detector</title>
    </head>
    <body style="font-family: Arial; text-align:center; margin-top:50px;">
        <h1>📰 Fake News Detection</h1>
        <form action="/predict" method="post">
            <textarea name="news" rows="8" cols="70" 
            placeholder="Paste your news text here..." required></textarea><br><br>
            <button type="submit" style="padding:10px 20px;">Check News</button>
        </form>
    </body>
    </html>
    """

# ==============================
# PREDICT ROUTE
# ==============================
@app.route("/predict", methods=["POST"])
def predict_news():
    if model is None:
        return "<h2>❌ Model not loaded. Check terminal error.</h2>"

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
    <body style="font-family: Arial; text-align:center; margin-top:50px;">
        <h1 style="color:{color};">{output}</h1>
        <br>
        <a href="/">🔙 Try Again</a>
    </body>
    </html>
    """

# ==============================
# RUN SERVER (FIXED)
# ==============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)