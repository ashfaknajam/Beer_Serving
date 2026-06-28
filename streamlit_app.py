import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path


@st.cache_data
def load_data(path="beer-servings.csv.csv"):
    df = pd.read_csv(path)
    return df


@st.cache_data
def train_model(df):
    target = "total_litres_of_pure_alcohol"
    features = ["beer_servings", "spirit_servings", "wine_servings", "continent"]
    X = df[features].copy()
    y = df[target].copy()

    numeric_features = ["beer_servings", "spirit_servings", "wine_servings"]
    numeric_transformer = SimpleImputer(strategy="median")

    categorical_features = ["continent"]
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))])

    # drop rows where target is missing
    mask = y.notna()
    model.fit(X[mask], y[mask])
    return model


def main():
    st.set_page_config(page_title="Beer Servings Alcohol Predictor", page_icon="🍺")
    st.title("Beer Servings — Alcohol Consumption Predictor")

    # welcome image: display local welcome.jpg in repo root only (fallback to remote if missing)
    local_img = Path("welcome.jpg")
    if local_img.exists():
        st.image(str(local_img), caption="Welcome — Predict total litres of pure alcohol", width=700)
    else:
        st.image(
            "https://images.unsplash.com/photo-1517677208171-0bc6725a3e60?auto=format&fit=crop&w=1200&q=60",
            caption="Welcome — Predict total litres of pure alcohol",
            width=700,
        )

    df = load_data()

    if st.checkbox("Show raw dataset (first 20 rows)"):
        st.dataframe(df.head(20))

    model = train_model(df)

    st.header("Enter country alcohol serving data")

    col1, col2, col3 = st.columns(3)
    with col1:
        beer = st.number_input("Beer servings", min_value=0.0, value=float(np.nanmedian(df["beer_servings"].dropna())))
    with col2:
        spirit = st.number_input("Spirit servings", min_value=0.0, value=float(np.nanmedian(df["spirit_servings"].dropna())))
    with col3:
        wine = st.number_input("Wine servings", min_value=0.0, value=float(np.nanmedian(df["wine_servings"].dropna())))

    continents = list(df["continent"].dropna().unique())
    continents = [c for c in continents if str(c) != "nan"]
    continents.insert(0, "Unknown")
    continent = st.selectbox("Continent", continents)

    if st.button("Predict total litres of pure alcohol"):
        input_df = pd.DataFrame(
            [{"beer_servings": beer, "spirit_servings": spirit, "wine_servings": wine, "continent": continent}]
        )
        pred = model.predict(input_df)[0]
        st.success(f"Predicted total litres of pure alcohol: {pred:.2f}")

        st.markdown("---")
        st.write("Model details: RandomForestRegressor trained on available countries.")


if __name__ == "__main__":
    main()
