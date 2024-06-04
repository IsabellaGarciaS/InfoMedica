from flask import Flask, request, jsonify, send_from_directory
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Cargar el modelo
with open('model/best_model.pkl', 'rb') as f:
    modelo = pickle.load(f)

# Función para preprocesar los datos del paciente
def preprocess_data(df):
    # Renombrar columnas
    df.columns = [col.replace(' ', '_').replace('\n', '') for col in df.columns]

    # Aplicar one-hot encoding a las columnas categóricas
    df = pd.get_dummies(df, columns=['DESCRIPCION'])

    # Mapear la columna 'Grupo_Edad'
    grupo_edad_map = {
        '0': 0, '1-4': 2.5, '5-9': 7, '10-14': 12, '15-19': 17,
        '20-24': 22, '25-29': 27, '30-34': 32, '35-39': 37,
        '40-44': 42, '45-49': 47, '50-54': 52, '55-59': 57,
        '60-64': 62, '65-69': 67, '70-74': 72, '75-79': 77,
        '80-84': 82, '85-89': 87, '> 90': 95
    }
    df['Grupo_Edad'] = df['Grupo_Edad'].map(grupo_edad_map)

    # Eliminar columnas innecesarias
    columnas_eliminar = [
        'COD', 'Año', 'Mes', 'Paciente_Tipo_Identificacion', 'Nro_Atencion', 
        'Paciente_Entidad_Responsable_Pago', 'Paciente_Modalidad_Contrato', 
        'Paciente_Regimen_Afiliacion', 'Urg_Fecha_(Camara)', 'Urg_Fecha_Ingreso', 
        'Urg_Fecha_Triaje', 'Urg_Fecha_Consulta_F3', 'Dias_(Camara_-_F3)', 
        'Horas_(Camara_-_F3)', 'Urg_Demora1_Consulta_(Minutos)', 
        'Dias_(Camara_-_Triaje)', 'Urg_Demora_Triaje_(Minutos)', 
        'Dias_(Triaje_-_F3)', 'Horas_(Triaje_-_F3)', 'Urg_Demora2_Consulta_(Minutos)', 
        'Profesional_Especialidad', 'Dx_Principal_Tipo', 'Dx_Principal', 
        'Dx_Principal_Capitulo', 'Dx_Principal.1', 'Dx_Relacionado1', 'Dx_Relacionado2', 
        'Dx_Relacionado3', 'Paciente_Edad', 'Unidad', 'Grupo_Edad', 'Paciente_Sexo', 
        'Profesional_Identificacion', 'Dias', 'Horas', 'Grupo_Poblacional', 
        'Pertenencia_Etnica', 'ALTO_COSTO'
    ]
    df = df.drop(columns=columnas_eliminar, errors='ignore')

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
    output = prediction[0]
    output = str(output)

    # Devolver la predicción como JSON
    return jsonify({'prediction': output})

@app.route('/')
def serve_prediction_page():
    return send_from_directory('static', 'prediction.html')

# Ejecutar la aplicación Flask
if __name__ == '__main__':
    app.run(debug=True)
