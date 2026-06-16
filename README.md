# diabetes-prediction-ml
AI-Powered Diabetes Prediction System using SVM and Streamlit
# 🏥 Diabetes Prediction System
### AI-Powered Risk Assessment Tool using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SVM](https://img.shields.io/badge/Algorithm-SVM-green?style=for-the-badge)

---

## 📸 Screenshots

### 🏠 Home Screen — Model Info
![Home Screen](Screenshot (245).png)

### ✅ Low Risk Result (18.5%)
![Low Risk](Screenshot (244).png)

### 🟡 Moderate Risk Result (60.1%)
![Moderate Risk](Screenshot (246).png)

---

## 📌 Project Overview

A **Machine Learning web application** that predicts the risk of diabetes based on patient medical measurements. Built using **Support Vector Machine (SVM)** and deployed with **Streamlit**.

> ⚠️ For **educational purposes only** — not a replacement for professional medical advice.

---

## 🎯 Features

- ✅ Real-time diabetes risk prediction
- ✅ Interactive gauge chart (0–100% risk level)
- ✅ Risk Factor Analysis (Glucose, BMI, Age, Blood Pressure)
- ✅ Positive Health Indicators
- ✅ Personalized health recommendations
- ✅ Clean dark-themed responsive UI

---

## 🧠 Machine Learning Details

| Property | Details |
|----------|---------|
| Algorithm | Support Vector Machine (SVM) |
| Kernel | RBF (Radial Basis Function) |
| Dataset | Pima Indians Diabetes Dataset |
| Total Samples | 768 |
| Train / Test Split | 80% / 20% |
| Accuracy | ~78% |
| Scaling | StandardScaler |

---

## 📊 Input Features

| Feature | Description | Range |
|---------|-------------|-------|
| Age | Patient age | 21–100 |
| Pregnancies | Number of pregnancies | 0–20 |
| Glucose | Plasma glucose (mg/dL) | 0–200 |
| Blood Pressure | Diastolic BP (mm Hg) | 0–130 |
| Skin Thickness | Triceps skinfold (mm) | 0–100 |
| Insulin | Serum insulin (mu U/ml) | 0–900 |
| BMI | Body Mass Index | 10–70 |
| Diabetes Pedigree | Genetic predisposition score | 0.0–2.5 |

---

## 🚀 How to Run Locally

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/diabetes-prediction-ml.git
cd diabetes-prediction-ml
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Train the model
```bash
python diabetes_prediction.py
```
> This generates `diabetes_model.pkl` and `scaler_svm.pkl` automatically

### Step 4 — Run the app
```bash
python -m streamlit run app.py
```

### Step 5 — Open browser
```
http://localhost:8501
```

---

## 📁 Project Structure

```
diabetes-prediction-ml/
│
├── app.py                    # Streamlit web app
├── diabetes_prediction.py    # Model training script
├── diabetes.csv              # Dataset
├── requirements.txt          # Dependencies
├── images/                   # Screenshots folder
│   ├── screenshot_home.png
│   ├── screenshot_low_risk.png
│   └── screenshot_moderate.png
└── README.md                 # Documentation
```

> 📝 `diabetes_model.pkl` and `scaler_svm.pkl` are auto-generated — not uploaded to GitHub

---

## 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
joblib
plotly
```

---

## 🔮 Future Improvements

- [ ] Add Random Forest & XGBoost models
- [ ] Model accuracy comparison dashboard
- [ ] Bulk CSV upload for multiple patients
- [ ] Deploy on Streamlit Cloud (free)
- [ ] Dataset visualization page

---

## 👨‍💻 Author

**Shewale Rohit** — ML Intern Project  
Built with ❤️ using Python, Scikit-learn & Streamlit

---

## 📜 License

Open source under [MIT License](LICENSE)

---

> 💡 **Deploy for FREE:** [share.streamlit.io](https://share.streamlit.io)
