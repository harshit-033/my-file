import pickle
import streamlit as st
import pandas as pd
import numpy as np
import sklearn

model = pickle.load(open('lr.pkl','rb'))

#first web

st.title("Scikit_learn first model with linear regression")

tv=st.text_input("Enter tv sales")
radio=st.text_input("Enter radio sales")
newspaper=st.text_input("Enter newspaper sales")





if st.button("Predict"):
    inp=np.array([[tv,radio,newspaper]], dtype=np.float64)

    prediction=model.predict(inp)
    st.write("predicted sales:- ",prediction)


