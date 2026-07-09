
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Insurance Predictor", page_icon="◉", layout="centered")

st.markdown("""
<style>
.block-container{max-width:760px;padding-top:2rem;padding-bottom:2rem}
h1,h2,h3{text-align:center}
.card{
padding:1.2rem;
border:1px solid rgba(120,120,120,.2);
border-radius:18px;
background:rgba(255,255,255,.03);
}
.small{color:#888;font-size:0.9rem;text-align:center;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("insurance_data.csv")

df = load_data()
X=df[["age"]]
y=df["bought_insurance"]

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42)

model=LogisticRegression()
model.fit(X_train,y_train)

st.title("Insurance Predictor")
st.markdown('<p class="small">Minimal. Clean. Focused.</p>',unsafe_allow_html=True)

st.markdown('<div class="card">',unsafe_allow_html=True)
age=st.slider("Age",18,80,30)

if st.button("Predict",use_container_width=True):
    x=np.array([[age]])
    prob=float(model.predict_proba(x)[0][1])
    pred=int(model.predict(x)[0])

    st.metric("Probability",f"{prob*100:.1f}%")
    if pred:
        st.success("Likely to purchase insurance.")
    else:
        st.info("Unlikely to purchase insurance.")
st.markdown("</div>",unsafe_allow_html=True)

c1,c2,c3=st.columns(3)
c1.metric("Samples",len(df))
c2.metric("Train Accuracy",f"{model.score(X_train,y_train)*100:.1f}%")
c3.metric("Test Accuracy",f"{model.score(X_test,y_test)*100:.1f}%")

with st.expander("Dataset"):
    st.dataframe(df,use_container_width=True)
