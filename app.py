from flask import Flask, request, render_template
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)
model = YOLO("models/best.pt")  # your trained model

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    img = Image.open(io.BytesIO(file.read()))
    results = model.predict(img)
    pred = results[0].names[results[0].probs.top1]
    return f"Prediction: {pred}"

if __name__ == "__main__":
    app.run(debug=True)

