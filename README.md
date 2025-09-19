# 🐄 **Image-Based Breed Recognition for Cattle and Buffaloes of India**

A deep learning project to recognize and classify Indian cattle and buffalo breeds from images using YOLOv8 and a Flask web app.

---

## ✨ Features
- Real-time detection & classification with YOLOv8  
- Upload or capture images via web frontend  
- Supports popular Indian cattle & buffalo breeds  
- Simple Flask backend for easy deployment  

---

## 📂 Project Structure
- `app.py` → Flask server  
- `models/` → trained weights (`best.pt`)  
- `datasets/` → training & validation data    
- `requirements.txt` → dependencies list  
- `README.md` → project documentation  

---

## ⚙️ Setup Instructions

1. **Clone the repo:**
 git clone https://github.com/username/Image-Based-Breed-Recognition-for-Cattle-and-Buffaloes-of-India.git DQAI_3(Clone the repo contents
&Put them directly inside a folder named DQAI_3)

 cd DQAI_3

   
**Create and activate virtual environment:**

python -m venv .venv

.venv\Scripts\activate      # Windows

source .venv/bin/activate   # Linux/Mac

->**Install dependencies:**
pip install -r requirements.txt

->Place your trained YOLO model weights (best.pt) inside the models/ folder.

**▶️ Run the App**

python app.py

Then open http://127.0.0.1:5000 in your browser.

Upload an image or capture from webcam

Get the breed prediction with bounding boxes

**🐃 Supported Breeds**
Gir,
Murrah,
Sahiwal,
Tharparkar,
Red Sindhi...
(Add more by retraining the model)

📜 License
This project is for educational and internal use.
Feel free to modify and extend as per your requirements.
