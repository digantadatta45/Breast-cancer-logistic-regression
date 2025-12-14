# 🎓 Breast Cancer Logistic Regression Streamlit App

This is an interactive **Streamlit app** built to demonstrate **logistic regression** on the **Breast Cancer Wisconsin dataset**. It predicts the probability of a tumor being **malignant** based on user-selected features.

## 🚀 Live Demo

[Open the app](https://breast-cancer-logistic-regression-wqeznareva59awg6nhxbtn.streamlit.app/)

---

## 💻 Features

- Dataset preview (first 10 rows shown for convenience)  
- Model evaluation metrics: **accuracy, precision, recall, F1-score**  
- Confusion matrix with **TP, FP, FN, TN** visualized  
- Feature coefficients and intercept display  
- **Interactive S-curve:** select a feature and input a value to see predicted probability  
- Scatter plot of actual data points  

---

## 🧠 How It Works

- Logistic Regression predicts **probability of malignancy** for each sample.  
- Probability is a number between **0 and 1**, showing the model's confidence.  
- Class prediction converts probability to 0 (benign) or 1 (malignant) using a threshold (0.5).  
- The **S-curve** visually shows how probability changes with the selected feature.  

---

## 🛠️ Run Locally

Download or clone this repository
Install dependencies: pip install -r requirements.txt
Run the app: streamlit run app.py
