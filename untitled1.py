import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Student Performance Analysis", layout="wide")
st.title("🎓 Student Performance Prediction & Analysis")

# 1. Load Dataset
# Note: Agar file ka naam 'processed_student_por.csv' hai toh wahi naam likhein
csv_filename = "student-por.csv"

try:
    df = pd.read_csv(csv_filename, sep=',')
except FileNotFoundError:
    try:
        df = pd.read_csv("processed_student_por.csv", sep=',')
    except FileNotFoundError:
        st.error(f"Dataset file '{csv_filename}' nahi mili! Apne GitHub repository mein CSV file zaroor upload karein.")
        st.stop()

st.subheader("1. Dataset Preview")
st.dataframe(df.head())

# 2. Dataset Information and Summary Statistics
st.subheader("2. Summary Statistics (Numerical)")
st.dataframe(df.describe())

st.write("**Missing Values per Column:**")
st.write(df.isnull().sum())

# 3. Data Visualization
st.subheader("3. Final Grade (G3) & Correlation")
sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Distribution of Final Grade (G3)
sns.histplot(df['G3'], bins=20, kde=True, ax=axes[0], color='skyblue')
axes[0].set_title('Distribution of Final Grades (G3)')
axes[0].set_xlabel('G3 Grade')
axes[0].set_ylabel('Count')

# Subplot 2: Correlation Heatmap
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=axes[1], cbar=True)
axes[1].set_title('Correlation Heatmap of Numerical Features')

plt.tight_layout()
st.pyplot(fig)

# 4. Pie Chart: Proportion of Students by School
st.subheader("4. Student Distribution by School (Pie Chart)")
school_counts = df['school'].value_counts()

fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
ax_pie.pie(
    school_counts,
    labels=['Gabriel Pereira (GP)', 'Mousinho da Silveira (MS)'],
    autopct='%1.1f%%',
    startangle=140,
    colors=['#ff9999', '#66b3ff'],
    explode=(0.05, 0),
    shadow=True
)
ax_pie.set_title('Distribution of Students by School')
st.pyplot(fig_pie)

# 5. Model Training (Pass/Fail Prediction)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.subheader("5. Machine Learning Model (Pass/Fail Prediction)")

# Target: Pass (1) if G3 >= 10, else Fail (0)
y = (df['G3'] >= 10).astype(int)
X = df.drop(columns=['G1', 'G2', 'G3'])
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.success(f"**Model Classification Accuracy: {accuracy * 100:.2f}%**")

# Classification Report
st.text("=== Classification Report ===")
st.text(classification_report(y_test, y_pred, target_names=['Fail', 'Pass']))

# 6. Confusion Matrix
st.subheader("6. Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)

fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Predicted Fail', 'Predicted Pass'],
    yticklabels=['Actual Fail', 'Actual Pass'],
    ax=ax_cm
)
ax_cm.set_title('Confusion Matrix')
ax_cm.set_ylabel('True Class')
ax_cm.set_xlabel('Predicted Class')
st.pyplot(fig_cm)
