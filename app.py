import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(
    page_title="Salary Prediction",
    page_icon="💼",
    layout="wide"
)

st.title("💼 Salary Prediction ")
st.write("Predict salary based on years of experience using Machine Learning.")

@st.cache_data
def load_data():
    return pd.read_csv("archive (1).zip",compression="zip")

try:
    df = load_data()
    df.columns = df.columns.str.strip()

    X = df[["YearsExperience"]]
    y = df["Salary"]

    model = LinearRegression()
    model.fit(X, y)

    st.subheader("📊 Dataset Overview")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Average Salary", f"₹{df['Salary'].mean():,.0f}")
    col3.metric("Average Experience", f"{df['YearsExperience'].mean():.1f} years")

    st.divider()
    st.subheader("🔮 Predict Your Salary")

    experience = st.number_input(
        "Enter years of experience",
        min_value=0.0,
        max_value=50.0,
        value=2.0,
        step=0.5
)

    if st.button("Predict Salary", type="primary"):
        prediction = model.predict(
             pd.DataFrame({"YearsExperience": [experience]})
        )[0]
    st.success(f"Estimated Salary: ₹{prediction:,.2f}")

    st.divider()
    st.subheader("📈 Experience vs Salary")
    st.scatter_chart(
        df,
        x="YearsExperience",
        y="Salary"
    )

    st.caption("Educational project: predictions are estimates, not guaranteed salaries.")

except Exception as e:
    st.error("Could not load the dataset. Please check the CSV file and column names.")
    st.write("Details:", str(e))
