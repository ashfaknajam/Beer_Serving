# Beer Servings — Streamlit Regression App

This repository contains a Streamlit web app that trains a regression model on the provided `beer-servings.csv.csv` dataset and predicts `total_litres_of_pure_alcohol` from user inputs.

Files:
- `streamlit_app.py` — Streamlit application (trains model on startup and serves predictions).
- `beer-servings.csv.csv` — dataset (already in repo).
- `requirements.txt` — Python dependencies.

Quick deploy to Streamlit Cloud

1. Push this repo to GitHub.
2. Go to https://streamlit.io/cloud and click "New app" → connect your GitHub repo → select `main` branch and `streamlit_app.py` as the main file → Deploy.

The app trains on the dataset at first start; no additional steps required.

Local test

Install dependencies and run locally:
```bash
python -m venv .venv
source .venv/bin/activate   # mac/linux
.venv\Scripts\activate    # windows
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Discord share template

Use the following message to share the public URL once the app is deployed.

```
Hey team — I deployed a simple ML demo that predicts `total_litres_of_pure_alcohol` from beer/spirit/wine servings.
Try it here: <YOUR_STREAMLIT_APP_URL>

Input values for `beer_servings`, `spirit_servings`, `wine_servings`, and `continent` and press Predict.
Feedback welcome! 🍺
```
