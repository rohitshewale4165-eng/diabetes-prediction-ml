import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import joblib

print("=" * 50)
print("  Diabetes Prediction Model - Training")
print("=" * 50)

# -----------------------------------------------
# Step 1: Dataset load karo (Pima Indians Diabetes)
# -----------------------------------------------
print("\n[1/5] Dataset load ho raha hai...")

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

columns = [
    'Pregnancies', 'Glucose', 'BloodPressure',
    'SkinThickness', 'Insulin', 'BMI',
    'DiabetesPedigreeFunction', 'Age', 'Outcome'
]

try:
    df = pd.read_csv(url, header=None, names=columns)
    print(f"    Dataset downloaded! Shape: {df.shape}")
except Exception as e:
    print(f"    Internet se download nahi hua, local try karta hoon...")
    # Agar internet nahi hai toh chhota sample data use karo
    # (Sirf fallback ke liye)
    np.random.seed(42)
    n = 768
    data = {
        'Pregnancies':              np.random.randint(0, 17, n),
        'Glucose':                  np.random.randint(0, 200, n),
        'BloodPressure':            np.random.randint(0, 122, n),
        'SkinThickness':            np.random.randint(0, 99, n),
        'Insulin':                  np.random.randint(0, 846, n),
        'BMI':                      np.round(np.random.uniform(0, 67, n), 1),
        'DiabetesPedigreeFunction': np.round(np.random.uniform(0.08, 2.42, n), 3),
        'Age':                      np.random.randint(21, 81, n),
        'Outcome':                  np.random.randint(0, 2, n),
    }
    df = pd.DataFrame(data)
    print(f"    Fallback data use kiya. Shape: {df.shape}")

# -----------------------------------------------
# Step 2: Data dekhte hain
# -----------------------------------------------
print("\n[2/5] Data analysis...")
print(f"    Total records  : {len(df)}")
print(f"    Diabetic (1)   : {df['Outcome'].sum()}")
print(f"    Non-Diabetic(0): {(df['Outcome'] == 0).sum()}")
print(f"    Missing values : {df.isnull().sum().sum()}")

# -----------------------------------------------
# Step 3: Features aur Target alag karo
# -----------------------------------------------
print("\n[3/5] Features prepare ho rahe hain...")

X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"    Training samples : {len(X_train)}")
print(f"    Testing samples  : {len(X_test)}")

# -----------------------------------------------
# Step 4: Scaling (Standardization)
# -----------------------------------------------
print("\n[4/5] Data scale ho raha hai...")

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std  = scaler.transform(X_test)

# Scaler save karo
joblib.dump(scaler, 'scaler_svm.pkl')
print("    scaler_svm.pkl saved!")

# -----------------------------------------------
# Step 5: SVM Model train karo
# -----------------------------------------------
print("\n[5/5] SVM Model train ho raha hai...")

model = svm.SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True,   # predict_proba ke liye
    random_state=42
)
model.fit(X_train_std, y_train)

# Accuracy check
train_acc = accuracy_score(y_train, model.predict(X_train_std))
test_acc  = accuracy_score(y_test,  model.predict(X_test_std))

print(f"    Training Accuracy : {train_acc * 100:.2f}%")
print(f"    Testing Accuracy  : {test_acc  * 100:.2f}%")

# Model save karo
joblib.dump(model, 'diabetes_model.pkl')
print("    diabetes_model.pkl saved!")

# -----------------------------------------------
# Done!
# -----------------------------------------------
print("\n" + "=" * 50)
print("  SUCCESS! Dono files ban gayi:")
print("    diabetes_model.pkl")
print("    scaler_svm.pkl")
print("\n  Ab yeh command run karo:")
print("    streamlit run app.py")
print("=" * 50)