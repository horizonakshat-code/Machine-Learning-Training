import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ------------------------------------
# PAGE CONFIG
# ------------------------------------

st.set_page_config(
    page_title="Employee Retention Predictor",
    page_icon="💼",
    layout="wide"
)

# ------------------------------------
# LOAD CSS
# ------------------------------------

try:
    with open("style.css") as css:
        st.markdown(
            f"<style>{css.read()}</style>",
            unsafe_allow_html=True
        )
except:
    pass

# ------------------------------------
# LOAD DATA
# ------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("HR_comma_sep.csv")

    df.drop_duplicates(inplace=True)

    return df

df = load_data()

# ------------------------------------
# TRAIN MODEL
# ------------------------------------

X = df[
    [
        "satisfaction_level",
        "average_montly_hours",
        "promotion_last_5years"
    ]
]

y = df["left"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = LogisticRegression()

model.fit(
    X_train,
    y_train
)

accuracy = accuracy_score(
    y_test,
    model.predict(X_test)
)
# ------------------------------------
# HERO SECTION
# ------------------------------------

st.markdown("""
<div class="hero">
    <h1>💼 Employee Retention Predictor</h1>
    <p>Predict whether an employee is likely to stay or leave the company using Machine Learning.</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------
# DASHBOARD
# ------------------------------------

st.markdown("## 📊 Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "👥 Employees",
        f"{len(df)}"
    )

with col2:
    st.metric(
        "🎯 Accuracy",
        f"{accuracy*100:.2f}%"
    )

with col3:
    st.metric(
        "🤖 Model",
        "Logistic Regression"
    )

st.write("")
st.write("")
# ------------------------------------
# EMPLOYEE INPUT FORM
# ------------------------------------

st.markdown("## 📝 Employee Information")

left_col, right_col = st.columns(2)

with left_col:

    satisfaction = st.slider(
        "😊 Satisfaction Level",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.01
    )

    monthly_hours = st.slider(
        "⏰ Average Monthly Hours",
        min_value=90,
        max_value=320,
        value=200
    )

with right_col:

    promotion = st.selectbox(
        "🏆 Promotion in Last 5 Years",
        ["No", "Yes"]
    )

promotion = 1 if promotion == "Yes" else 0

st.write("")

predict = st.button(
    "🚀 Predict Employee Retention",
    use_container_width=True
)
# ------------------------------------
# PREDICTION
# ------------------------------------

if predict:

    sample = pd.DataFrame(
        {
            "satisfaction_level": [satisfaction],
            "average_montly_hours": [monthly_hours],
            "promotion_last_5years": [promotion]
        }
    )

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0][1]

    st.write("")
    st.subheader("Prediction Result")

    st.progress(float(probability))

    st.metric(
        "Leave Probability",
        f"{probability*100:.2f}%"
    )

    if prediction == 1:

        st.error(
            "⚠️ This employee is likely to leave the company."
        )

    else:

        st.success(
            "✅ This employee is likely to stay with the company."
        )

# ------------------------------------
# DATASET PREVIEW
# ------------------------------------

st.write("")
st.write("")

st.subheader("📋 Dataset Preview")

st.dataframe(
    df.head(10),
    width="stretch"
)

# ------------------------------------
# DATASET INFORMATION
# ------------------------------------

with st.expander("📊 Dataset Statistics"):

    st.write(df.describe())

# ------------------------------------
# FOOTER
# ------------------------------------

st.write("")
st.markdown("---")

st.markdown(
    """
    <div class="footer">
        <h4>Employee Retention Prediction</h4>
        <p>Built with Streamlit • Scikit-Learn • Logistic Regression</p>
    </div>
    """,
    unsafe_allow_html=True
)
