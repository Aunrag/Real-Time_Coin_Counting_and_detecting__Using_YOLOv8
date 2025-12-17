# Real-Time Coin Counting Using YOLOv8

This project implements a **real-time coin counting and valuation system** using a **custom-trained YOLOv8 object detection model**. The system detects multiple coin denominations from a live camera feed and calculates the total monetary value in real time.

---

##  Features
- Custom-trained YOLOv8 model on a labeled coin dataset
- Real-time coin detection using webcam
- Automatic coin counting and total value calculation
- Bounding box visualization with denomination labels
- Works on images, videos, and live camera feed

---

##  Tech Stack
- Python
- Ultralytics YOLOv8
- OpenCV
- Google Colab (for training)
- CUDA GPU (recommended for training)

---

##  Project Structure
  ─ 2_models/coin_yolov8_last.pt 
  
  ─ notebooks/coin_training.ipynb
  
  ─ live_coin_counter.py
  
  ─ requirements.txt
  
  ─ README.md


---

##  Model Training 
### Steps to Train the Model:
1. Open **`coin_training.ipynb`** in Google Colab
2. Enable GPU:
   - `Runtime` → `Change runtime type` → Select **GPU**
3. Install dependencies inside Colab:
   ```bash
   pip install ultralytics
   ```
4.Upload your custom coin dataset (images + labels)

5.Run all cells to train the YOLOv8 model

6.Download the trained model (coin_yolov8_last.pt)

7.Place it inside the 2_models folder

---
# Running Live Coin Counter (Webcam)
 Install Dependencies 
```bash
pip install ultralytics opencv-python
```
 Run the Application 
 ```python
python live_coin_counter.py
```
Press q to exit the live camera window.

