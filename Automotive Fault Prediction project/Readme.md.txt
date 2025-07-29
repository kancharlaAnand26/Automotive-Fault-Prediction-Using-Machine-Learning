# 🚗 Automotive Fault Prediction Using Machine Learning

This project predicts automotive faults like engine overheating or brake failure using sensor data and machine learning. It simulates predictive maintenance systems used in connected vehicles.

## 🔧 Tools Used
- Python, Scikit-learn, Pandas, NumPy
- Streamlit (optional UI)
- Matplotlib/Seaborn for visualization

## 📊 Dataset
Synthetic dataset simulating car sensors:
- Speed, Engine Temp, Oil Pressure, Vibration, Brake Pressure, etc.
- Target: Fault types like No Fault, Engine Overheat, Brake Issue.

## ⚙️ How It Works
1. Clean and preprocess data
2. Train model using Random Forest Classifier
3. Evaluate model using accuracy, precision, recall
4. Optional Streamlit UI to predict faults in real time

## 🧪 Accuracy
Achieved 91% F1-Score with Random Forest.

## ▶️ Demo
![Demo Video](demo/demo.mp4)

## 🧠 Architecture
![Architecture](architecture.png)

## 💻 How to Run

```bash
git clone https://github.com/yourname/automotive-fault-prediction
cd automotive-fault-prediction
pip install -r requirements.txt
jupyter notebook
