from flask import Flask, request, jsonify, send_from_directory
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Carga el modelo
with open('model/best_model.pkl', 'rb') as f:
    modelo = pickle.load(f)

# Función para preprocesar los datos del paciente
def preprocess_data(df):
    # Eliminar columnas no necesarias
    # df = df.drop(columns=['COD','Año','Mes','Paciente Tipo Identificacion',  'Nro Atencion', 'Paciente Entidad Responsable Pago',
    #                          'Paciente Modalidad Contrato','Paciente Regimen Afiliacion', 'Urg Fecha (Camara)','Urg Fecha Ingreso',
    #                          'Urg Fecha Triaje','Urg Fecha Consulta F3' , 'Dias (Camara - F3)','Horas (Camara - F3)',
    #                          'Urg Demora1 Consulta (Minutos)', 'Dias (Camara - Triaje)', 'Urg Demora Triaje (Minutos)', 'Dias (Triaje - F3)',
    #                          'Horas (Triaje - F3)','Urg Demora2 Consulta (Minutos)','Profesional Especialidad','Dx Principal Tipo ',
    #                          'Dx Principal', 'Dx Principal Capitulo', 'Dx Principal.1','Dx Relacionado1','Dx Relacionado2','Dx Relacionado3', 'Paciente Edad',
    #                          'Unidad', 'Grupo Edad', 'Paciente Sexo',  'Profesional Identificacion', 'Dias', 'Horas', 'Grupo Poblacional', 'Pertenencia Etnica',
    #                          'ALTO COSTO'])

    # Codificar las columnas categóricas usando One-Hot Encoding
    df = pd.get_dummies(df)

    return df

# Definir una ruta para hacer predicciones
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener el archivo Excel del cuerpo de la solicitud
    file = request.files['file']

    # Cargar el archivo Excel en un DataFrame
    df_paciente = pd.read_excel(file)

    # Preprocesar los datos del paciente
    X_new = preprocess_data(df_paciente)

    # Hacer la predicción con el modelo
    prediction = modelo.predict(X_new)
    output=prediction[0]
    output=str(output)

    # Devolver la predicción como JSON
    return jsonify({'prediction': output})

@app.route('/')
def serve_prediction_page():
    return send_from_directory('static', 'prediction.html')

# Ejecutar la aplicación Flask
if __name__ == '__main__':
    app.run(debug=True)