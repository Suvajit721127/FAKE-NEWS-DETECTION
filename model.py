import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==============================
# DEVICE
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==============================
# LOAD DATA
# ==============================
def load_data():
    path = "dataset-merged.csv (1).zip"

    df = pd.read_csv(path, compression="zip")
    df = df[["text", "label"]].dropna()

    df["text"] = df["text"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna()

    print("Dataset loaded:", len(df))
    return df

# ==============================
# LOAD HINVEC MODEL
# ==============================
def load_hinvec():
    model_name = "Sailesh97/Hinvec"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    model.to(device)
    model.eval()

    return tokenizer, model

# ==============================
# EMBEDDING FUNCTION
# ==============================
def get_hinvec_embedding(text, tokenizer, model):
    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=32
    )

    input_ids = tokens["input_ids"].to(device)

    with torch.no_grad():
        token_embeddings = model.get_input_embeddings()(input_ids)

    embedding = token_embeddings.mean(dim=1)

    return embedding.squeeze().cpu().numpy()

# ==============================
# GENERATE EMBEDDINGS
# ==============================
def generate_embeddings(df, tokenizer, model):
    X = []
    y = df["label"].values

    print("Generating embeddings...")

    for text in df["text"]:
        emb = get_hinvec_embedding(str(text), tokenizer, model)
        X.append(emb)

    X = np.array(X)

    print("Embedding shape:", X.shape)
    return X, y

# ==============================
# RNN MODEL
# ==============================
class RNNClassifier(nn.Module):
    def __init__(self, input_size=2048, hidden_size=128):
        super().__init__()
        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            batch_first=True,
            nonlinearity='tanh'
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, h_n = self.rnn(x)
        out = self.fc(h_n[-1])
        return out.squeeze(1)

# ==============================
# TRAIN + SAVE MODEL
# ==============================
def train_and_save_model():
    df = load_data()
    tokenizer, hinvec_model = load_hinvec()

    X, y = generate_embeddings(df, tokenizer, hinvec_model)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32).unsqueeze(1)

    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test,  dtype=torch.float32)

    model = RNNClassifier(input_size=X.shape[1]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # TRAIN LOOP
    for epoch in range(10):
        model.train()
        total_loss = 0

        for i in range(len(X_train_t)):
            xb = X_train_t[i:i+1].to(device)
            yb = y_train_t[i:i+1].to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/10 | Loss: {total_loss/len(X_train_t):.4f}")

    # TEST
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t.to(device))
        y_pred = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()

    print("Accuracy:", accuracy_score(y_test, y_pred))

    # ✅ SAVE MODEL
    torch.save(model.state_dict(), "model.pth")
    print("Model saved as model.pth")

# ==============================
# LOAD TRAINED MODEL
# ==============================
def load_trained_model(input_size=2048):
    model = RNNClassifier(input_size=input_size)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

# ==============================
# PREDICT FUNCTION (FOR WEB)
# ==============================
def predict(text, model, tokenizer, hinvec_model):
    emb = get_hinvec_embedding(text, tokenizer, hinvec_model)

    emb_t = torch.tensor(emb, dtype=torch.float32)\
        .unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(emb_t)
        prediction = int(torch.sigmoid(logit).item() >= 0.5)

    return prediction

# ==============================
# MAIN (ONLY TRAIN ONCE)
# ==============================
if __name__ == "__main__":
    train_and_save_model()