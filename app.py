import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

st.set_page_config(page_title="⚡ Energy Data Prediction", page_icon="⚡", layout="wide")
st.title("⚡ Energy Data Visualization and Random Forest Prediction")
st.caption("Predicts overall load behavior (aggregated numeric response) using Random Forest")

DATA_PATH = "smart_grid_dataset.csv"

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"❌ `{DATA_PATH}` not found. Please place it in the same folder as this file.")
    st.stop()

st.header("📄 Dataset Preview")
st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
st.dataframe(df.head(20), use_container_width=True)

st.header("📊 Summary Statistics")
st.write(df.describe())

st.header("📈 Visualizations")

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
if not numeric_cols:
    st.warning("No numeric columns found in the dataset.")
    st.stop()

col_choice = numeric_cols[0]
st.subheader(f"Histogram of {col_choice}")
fig, ax = plt.subplots()
sns.histplot(df[col_choice], bins=30, kde=True, ax=ax, color="skyblue")
st.pyplot(fig)

st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(8, 6))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

new = {}
for col in df.columns:
    if 'Temperature' in col and '°' not in col:
        new[col] = col.replace('(\xC2\xB0C)', '(°C)')
if new:
    df.rename(columns=new, inplace=True)

removed_cols = ['Timestamp', 'ID'] if 'ID' in df.columns else ['Timestamp']
df = df.drop(columns=[c for c in removed_cols if c in df.columns], errors='ignore')

df = df.dropna()

st.success(f"✅ Data cleaned successfully! New shape: {df.shape}")

target = 'Load' if 'Load' in df.columns else df.columns[-1]
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.header("📊 Actual vs Predicted Load (Scatter Plot)")
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_test, y_pred, alpha=0.6, color="green")
ax.set_xlabel("Actual Load")
ax.set_ylabel("Actual vs Predicted Load")
ax.set_title("Actual vs Predicted Load (Notebook Style)")
st.pyplot(fig)
