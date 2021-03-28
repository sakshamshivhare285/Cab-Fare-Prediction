import pickle
import streamlit as st
import pandas as pd
import numpy as np

pickle_in=open('predictor.pkl', mode='rb')
predictor=pickle.load(pickle_in)

def run():       
    add_selectbox=st.sidebar.selectbox(
      "How would like to get the predictions?",
      ('Realtime','Batch'))
    st.sidebar.info('This application helps to predict cab fare')
    if add_selectbox=='Realtime':
        passenger=st.number_input('passengers', min_value=1, max_value=6, value=1)
        hour=st.number_input('hour', min_value=0, max_value=23, value=0)
        mins=st.number_input('mins', min_value=0, max_value=59, value=0)
        day=st.number_input('day', min_value=0, max_value=6, value=0)
        distance=st.number_input('distance',min_value=0, max_value=1000, value=0)
        year=st.number_input('year', min_value=2009, max_value=2015, value=2009)
        date=st.number_input('date', min_value=1, max_value=31, value=1)
        output=''
        input_dict={'passenger':passenger, 'hour':hour,'mins':mins,'day':day,
                'distance':distance,'year':year,'date':date}
        input_df=pd.DataFrame([input_dict])
        if st.button("predict"):
            output=predictor(input_df=input_df)
            output=float(output)
    if add_selectbox=='Batch':
        file_upload=st.file_uploader("upload the csv file", type=['csv'])
        if file_upload is not None:
            data=pd.read_csv(file_upload)
            predictions=predict_model(estimator=predictor, data=data)
            st.write(predictions)
run()
