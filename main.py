import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as ff
from save_model import pickle
import seaborn as sns
# from sklearn.linear_model import LogisticRegression



header = st.container()
dataset = st.container()
datavisual = st.container()
modelTraining = st.container()
footnote= st.container()

@st.cache
def getdata(filename):
    voice_data = pd.read_csv('voice.csv')
    return voice_data

with header:
    st.title('Gender Recognition through Voice')
  


with dataset:
    st.header('Voice acoustic features extracted ')

    voice_data = getdata('voice.csv')

    if st.checkbox('Preview Dataset'):
         st.write(voice_data.head(5))

with datavisual:
    st.header('Exploring the dataset to show visuals')

    if st.checkbox('Visuals'):
        if st.button('Show Histogram'):   
            fig = ff.histogram(voice_data,x='meanfreq',y='median',color='label')
            st.plotly_chart(fig, use_container_width=True)
            fig = ff.histogram(voice_data,x='sd',y='kurt',color='label')
            st.plotly_chart(fig, use_container_width=True)
        if st.button('Show Line chart'):
            df = pd.DataFrame(voice_data[:200],columns=['IQR','Q25','Q75'])
            st.line_chart(df)
        if st.button('Show Area Chart'):
            chart_data = pd.DataFrame(voice_data[:100],columns=['centroid','mode'])
            st.area_chart(chart_data)
        if st.button('Show Bar chart'):
            chart = pd.DataFrame(voice_data[:50],columns=['sfm','sp.ent'])
            st.bar_chart(chart)
            # st.bar_chart(voice_data[''])
        if st.button('Show heatmap'):
            fig, ax = plt.subplots()
            sns.heatmap(voice_data[:20].corr(), ax=ax)
            st.write(fig)

with modelTraining:
    # st.header('Time to train the model')
    pickle_in = open('logit.pkl', 'rb')
    classifier = pickle.load(pickle_in)

    st.sidebar.header('Gender Recognition')
    select = st.sidebar.selectbox('Select Form', ['Form 1'], key='1')
    if not st.sidebar.checkbox("Hide", True, key='1'):
        st.title('Gender Recogition with voice features')
        sd = st.number_input("sd value:")
        Q75 = st.number_input("Q75 value:")
        IQR = st.number_input("IQR value:")
        skew = st.number_input("skew value:")
        kurt = st.number_input("kurt value:")
        spent = st.number_input("sp.ent value:")
        sfm = st.number_input("sfm value")
        modindx = st.number_input("modindx value:")

    submit = st.button('Predict')
    if submit:
            prediction = classifier.predict([[sd,Q75,IQR,skew,kurt,spent,sfm,modindx]])
            if prediction == 0:
                st.write('Gender is a Female')
            else:
                st.write('Gender is a Male')