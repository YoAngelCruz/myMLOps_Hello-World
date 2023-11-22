import json
import streamlit as st
import requests

SERVER_URL = 'https://bug-model-service-yoangelcruz.cloud.okteto.net/v1/models/bug-model:predict'

def make_prediction(inputs):
    predict_request = {'instances': inputs}
    response = requests.post(SERVER_URL, json=predict_request)
    response.raise_for_status()
    prediction = response.json()
    return prediction

def main():
    st.title('Predicción de Errores en Software')

    code_size = st.number_input('Ingrese el número de líneas de código:', min_value=0, step=1)
    complexity = st.number_input('Ingrese la complejidad del código:', min_value=0, step=1)

    if st.button('Predecir'):
        prediction = make_prediction([[code_size, complexity]])
        predicted_errors = prediction['predictions'][0][0]
        st.write(f'Número estimado de errores para un código de {code_size} líneas y complejidad {complexity}: {predicted_errors}')

if __name__ == '__main__':
    main()
