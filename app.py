# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score

st.set_page_config(page_title="Breast Cancer Logistic Regression", layout="wide")


# Load dataset
data = load_breast_cancer()
X_full = data.data
y_full = data.target
feature_names = data.feature_names

df = pd.DataFrame(X_full, columns=feature_names)
df["target"] = y_full

st.title("Breast Cancer Logistic Regression Demo")

# Dataset overview
st.subheader("Dataset Overview")
st.write(f"Number of samples: {X_full.shape[0]}, Number of features: {X_full.shape[1]}")
st.dataframe(df.head(10))
st.info("Note: Only the first 10 rows are shown here for preview. The model is trained on the entire dataset.")
st.write("Target value counts:")
st.write(df["target"].value_counts())



# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)


# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train logistic regression
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_scaled, y_train)


# Model evaluation
st.subheader("Model Evaluation")

y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

st.write(f"**Accuracy:** {accuracy:.4f} – Overall correctness of predictions")
st.write(f"**Precision:** {precision:.4f} – Of all predicted malignant cases, how many were actually malignant")
st.write(f"**Recall (Sensitivity):** {recall:.4f} – Of all actual malignant cases, how many were correctly detected")
st.write(f"**F1-score:** {f1:.4f} – Harmonic mean of precision and recall")

st.write("**Classification Report:**")
st.text(classification_report(y_test, y_pred))

# Confusion matrix as graph with TP, FP, FN, TN
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5,5))
im = ax.imshow(cm, cmap="Blues")

# Labels
labels = np.array([["TN", "FP"], ["FN", "TP"]])
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{labels[i,j]}\n{cm[i,j]}", ha="center", va="center", color="black", fontsize=12)

ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_xticklabels(data.target_names)
ax.set_yticklabels(data.target_names)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix with TP, FP, FN, TN")
fig.colorbar(im)
st.pyplot(fig)

st.write("""
**Legend:**  
- **TN (True Negative):** Correctly predicted benign  
- **FP (False Positive):** Predicted malignant but actually benign  
- **FN (False Negative):** Predicted benign but actually malignant  
- **TP (True Positive):** Correctly predicted malignant
""")


# Show model parameters
st.subheader("Model Parameters")
coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": clf.coef_[0]
}).sort_values(by="Coefficient", key=abs, ascending=False)
st.dataframe(coef_df)
st.write(f"Intercept (b): {clf.intercept_[0]:.4f}")


# Interactive S-curve for selected feature
st.subheader("Interactive Logistic Regression S-Curve")
selected_feature = st.selectbox("Select feature to visualize:", feature_names)
feature_idx = np.where(feature_names == selected_feature)[0][0]

# Slider input
min_val = float(X_full[:, feature_idx].min())
max_val = float(X_full[:, feature_idx].max())
user_input = st.slider(f"Select {selected_feature} value:", min_value=min_val, max_value=max_val, value=(min_val+max_val)/2)

# Prepare grid
grid = np.linspace(min_val, max_val, 200)
X_grid = np.tile(X_full.mean(axis=0), (200,1))
X_grid[:, feature_idx] = grid
X_grid_scaled = scaler.transform(X_grid)
prob_grid = clf.predict_proba(X_grid_scaled)[:,1]

# User input probability
X_user = X_full.mean(axis=0).reshape(1,-1)
X_user[0, feature_idx] = user_input
X_user_scaled = scaler.transform(X_user)
user_prob = clf.predict_proba(X_user_scaled)[:,1][0]

# Plot S-curve
fig2, ax2 = plt.subplots(figsize=(7,4))
ax2.scatter(X_full[:, feature_idx], y_full, alpha=0.3, label="Actual data")
ax2.plot(grid, prob_grid, color="red", label="Predicted probability (S-curve)")
ax2.axvline(user_input, color="green", linestyle="--", label=f"Your input: {user_input:.2f}")
ax2.scatter(user_input, user_prob, color="green", s=100, zorder=5, label=f"P(Malignant)={user_prob:.2f}")
ax2.set_xlabel(selected_feature)
ax2.set_ylabel("Probability of Malignant")
ax2.set_title(f"Logistic Regression Curve for {selected_feature}")
ax2.grid(True)
ax2.legend()
st.pyplot(fig2)
