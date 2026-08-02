# ❤️ Heart Disease Prediction App

A Streamlit web app that predicts a patient's risk of heart disease using a
K-Nearest Neighbors (KNN) classifier trained on clinical health data.

</div>

---

## Demo

Run locally with `streamlit run app.py` — see [Setup](#setup) below.

</div>

---

## Features

- Simple form-based UI to enter patient details (age, sex, chest pain type,
  blood pressure, cholesterol, ECG results, exercise-induced angina, etc.)
- Predicts **low** or **high** risk of heart disease on submission
- Uses a pre-trained scikit-learn KNN model with feature scaling
- Correctly one-hot encodes categorical inputs to match the columns the
  model was trained on

</div>

---

## Project Structure

```
heart-disease-prediction/
├── app.py                     # Streamlit application
├── model/
│   ├── knn_heart_model.pkl    # Trained KNN classifier
│   ├── heart_scaler.pkl       # Fitted StandardScaler
│   └── heart_columns.pkl      # Expected input columns (for one-hot alignment)
├── data/
│   ├── heart.csv              # Training dataset (918 records)
│   └── README.md
├── notebooks/
│   └── heart_disease.ipynb    # EDA, model comparison, and training
├── requirements.txt
├── .gitignore
└── README.md
```
---

</div>

---

## 🛠️ Tech Stack

<div align="center">
  
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter_Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

</div>

---

## Dataset

The model is trained on the [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
(also available as `data/heart.csv` in this repo), which combines five
public heart-disease datasets (Cleveland, Hungarian, Switzerland, Long Beach VA,
and Stalog) into a single set of 918 patient records with 11 clinical features
and a binary `HeartDisease` target.

| Feature | Description |
|---|---|
| Age | Age of the patient (years) |
| Sex | M / F |
| ChestPainType | TA, ATA, NAP, ASY |
| RestingBP | Resting blood pressure (mm Hg) |
| Cholesterol | Serum cholesterol (mg/dL) |
| FastingBS | 1 if fasting blood sugar > 120 mg/dL, else 0 |
| RestingECG | Normal, ST, LVH |
| MaxHR | Maximum heart rate achieved |
| ExerciseAngina | Y / N |
| Oldpeak | ST depression induced by exercise |
| ST_Slope | Up, Flat, Down |
| HeartDisease | 1 = heart disease, 0 = normal (target) |

</div>

---

## Model

- **Algorithm:** K-Nearest Neighbors (`scikit-learn`) — selected after
  comparing several classifiers (see `notebooks/heart_disease.ipynb`)
- **Preprocessing:** categorical features one-hot encoded (`pd.get_dummies`,
  `drop_first=True`), numerical features scaled with `StandardScaler`
- Trained artifacts are checked into `model/` so the app runs out of the box
  without retraining

</div>

---

### Model comparison (held-out test set)

| Model | Accuracy | F1 Score |
|---|---|---|
| **KNN (used in app)** | **0.886** | **0.899** |
| Logistic Regression | 0.875 | 0.888 |
| Naive Bayes | 0.870 | 0.879 |
| SVM (RBF Kernel) | 0.864 | 0.880 |
| Decision Tree | 0.766 | 0.782 |

KNN gave the best accuracy and F1 score of the models compared, which is why
it was chosen for the deployed app.

</div>

---

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

### Reproducing the training notebook (optional)

```bash
pip install -r requirements-dev.txt
jupyter notebook notebooks/heart_disease.ipynb
```

</div>

---

## Deploying

Deploy for free on [Streamlit Community Cloud](https://streamlit.io/cloud):
1. Push this repo to GitHub.
2. Go to share.streamlit.io, connect your GitHub account, and select this repo.
3. Set the main file path to `app.py` and deploy.

</div>

---

## Roadmap / Ideas

- [ ] Add cross-validation and hyperparameter tuning for KNN (`n_neighbors`, weighting)
- [ ] Add input validation and unit tests
- [ ] Add a confusion matrix / ROC curve plot to the notebook and README
- [ ] Try ensemble methods (Random Forest, XGBoost) for comparison

</div>

---

## Disclaimer

This tool is for educational/demonstration purposes only and is **not** a
substitute for professional medical advice, diagnosis, or treatment.
