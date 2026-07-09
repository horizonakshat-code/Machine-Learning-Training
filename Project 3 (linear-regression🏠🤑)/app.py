
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Smart House Value Estimator",
    page_icon="🏡",
    layout="wide"
)

@st.cache_data
def load_dataset():
    return pd.read_csv("houseprice.csv")

@st.cache_resource
def build_model(data):
    features = data[["area"]]
    target = data["price"]
    regressor = LinearRegression()
    regressor.fit(features, target)
    return regressor

def main():
    st.title("🏡 Smart House Value Estimator")
    st.caption("Estimate property prices from house area using Linear Regression.")

    housing_data = load_dataset()
    price_model = build_model(housing_data)

    st.sidebar.header("Input")
    selected_area = st.sidebar.slider(
        "House Area (sq.ft)",
        min_value=int(housing_data["area"].min()),
        max_value=int(housing_data["area"].max()),
        value=int(housing_data["area"].median()),
        step=50,
    )

    left, right = st.columns([3, 2])

    with left:
        st.subheader("Dataset Preview")
        st.dataframe(housing_data, use_container_width=True)
        st.subheader("Area vs Price")
        st.scatter_chart(housing_data, x="area", y="price")

    with right:
        st.subheader("Prediction")
        estimated_price = float(price_model.predict([[selected_area]])[0])

        st.metric("Estimated Price", f"₹ {estimated_price:,.0f}")
        st.metric("Selected Area", f"{selected_area} sq.ft")

        with st.expander("Model Information"):
            st.write(f"Coefficient: {price_model.coef_[0]:.4f}")
            st.write(f"Intercept: {price_model.intercept_:.2f}")

        if st.button("Predict Again"):
            st.success(
                f"A house with an area of **{selected_area} sq.ft** "
                f"is predicted to cost **₹ {estimated_price:,.2f}**."
            )

if __name__ == "__main__":
    main()
