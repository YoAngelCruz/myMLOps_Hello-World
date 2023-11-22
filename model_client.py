import json
import streamlit as st
import requests

SERVER_URL = 'https://linear-model-service-yoangelcruz.cloud.okteto.net/v1/models/linear-model:predict'

def make_prediction(inputs):
   
    predict_request = {'instances': [inputs]}
    response = requests.post(SERVER_URL, json=predict_request)
    response.raise_for_status()
    prediction = response.json()
    return prediction

def main():
    st.title('Estimador del peso de un objeto basado en su masa')

   
    code_size = st.number_input('Ingrese la masa del objeto:', min_value=0.0, step=1.0)

   
    if st.button('Predecir'):
        prediction = make_prediction([code_size])
        st.write(f'Peso del objeto basado en la masa de {code_size}: {prediction["predictions"][0][0]}')

if __name__ == '__main__':
    main()