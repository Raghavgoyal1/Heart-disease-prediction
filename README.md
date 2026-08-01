# Heart Disease Prediction App

A Streamlit web app that predicts the risk of heart disease using a K-Nearest
Neighbors (KNN) classifier trained on patient health data.

## Features

- Simple form-based UI to enter patient details (age, sex, chest pain type,
  blood pressure, cholesterol, ECG results, etc.)
- Predicts **low** or **high** risk of heart disease on submission
- Uses a pre-trained scikit-learn KNN model with feature scaling

## Project Structure

```
heart-disease-prediction/
├── app.py                     # Streamlit application
├── model/
│   ├── knn_heart_model.pkl    # Trained KNN classifier
│   ├── heart_scaler.pkl       # Fitted StandardScaler
│   └── heart_columns.pkl      # Expected input columns (for one-hot alignment)
├── requirements.txt
├── .gitignore
└── README.md
```

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

## Deploying

This app can be deployed for free on [Streamlit Community Cloud](https://streamlit.io/cloud):
1. Push this repo to GitHub.
2. Go to share.streamlit.io, connect your GitHub account, and select this repo.
3. Set the main file path to `app.py` and deploy.

## Known Issue

The current input-encoding logic builds one-hot column names directly from
the raw dropdown labels (e.g. `"Sex_" + sex`), but the model was trained on
differently-named columns (e.g. `Sex_M`, `ExerciseAngina_Y`,
`RestingECG_ST`). As written, several fields silently fall back to 0 instead
of matching the trained columns. The app still runs without errors, but
predictions may not fully reflect the Sex, Exercise Angina, and some
Resting ECG inputs. This should be fixed by mapping UI labels to the exact
column suffixes found in `model/heart_columns.pkl`.

## Disclaimer

This tool is for educational/demonstration purposes only and is **not** a
substitute for professional medical advice, diagnosis, or treatment.
