import os
from flask import Flask, request, render_template, send_from_directory, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Asegurarse de que el directorio 'uploads' exista
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Directorio donde se guardará la imagen generada
STATIC_FOLDER = 'static'
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    dias_a_predecir = request.form.get('dias')  # Obtener el valor de los días a predecir
    if not dias_a_predecir or not dias_a_predecir.isdigit():
        return jsonify({'error': 'Debe ingresar un número válido de días a predecir'})

    dias_a_predecir = int(dias_a_predecir)

    # Guardar el archivo en el directorio 'uploads'
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Procesar el archivo CSV
    try:
        # Cargar los datos
        df = pd.read_csv(filepath)

        # Limpiar y preparar los datos
        df['Precio Interno '] = pd.to_numeric(df['Precio Interno ']
                                            .str.replace(r'[^0-9,]', '', regex=True)  # Remove *m² and $, except commas
                                            .str.replace(',', '.', regex=False))  # Commas with dots
        
        # Convertir la columna 'Fecha' a tipo datetime, forzando los valores incorrectos a NaT
        df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d-%b-%y', errors='coerce')

        # Filtrar filas con fechas NaT
        df = df.dropna(subset=['Fecha'])

        # Filtrar datos (2022-2024)
        data_filtered = df[(df['Fecha'].dt.year >= 2022) & (df['Fecha'].dt.year <= 2024)]
        
        # --------------------------------------------
        # Preparar los datos para el modelo
        SECUENCIA_DIAS = 5
        precios = df['Precio Interno '].values.reshape(-1,1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        precios_scaled = scaler.fit_transform(precios)

        X, y = [], []
        for i in range(SECUENCIA_DIAS, len(precios_scaled)):
            X.append(precios_scaled[i-SECUENCIA_DIAS:i, 0])
            y.append(precios_scaled[i, 0])

        X = np.array(X)
        y = np.array(y)

        # --------------------------------------------
        # Construir el modelo MLP
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(SECUENCIA_DIAS,)))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1))  # Salida única para regresión

        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse')

        # Callbacks para eficiencia en entrenamiento
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

        # Entrenamiento del modelo
        history = model.fit(X, y, validation_split=0.2, epochs=100, batch_size=16, callbacks=[early_stop, reduce_lr], verbose=1)

        # --------------------------------------------
        # Realizar predicciones futuras
        ultimos_dias = precios_scaled[-SECUENCIA_DIAS:]
        predicciones_futuras = []

        for _ in range(dias_a_predecir):
            proxima_prediccion = model.predict(ultimos_dias.reshape(1, -1))
            predicciones_futuras.append(proxima_prediccion[0, 0])
            ultimos_dias = np.append(ultimos_dias[1:], proxima_prediccion)  # Actualizar para la siguiente predicción

        predicciones_futuras_orig = scaler.inverse_transform(np.array(predicciones_futuras).reshape(-1, 1))

        # Crear la gráfica
        fechas_futuras = pd.date_range(start=df['Fecha'].iloc[-1] + pd.Timedelta(days=1), periods=dias_a_predecir)
        
        plt.figure(figsize=(10,6))
        plt.plot(df['Fecha'], df['Precio Interno '], label='Histórico')
        plt.plot(fechas_futuras, predicciones_futuras_orig, label='Predicción')
        plt.title('Predicción del Precio del Café para los próximos días')
        plt.xlabel('Fecha')
        plt.ylabel('Precio')
        plt.legend()

        # Guardar la gráfica
        graph_path = os.path.join(app.config['STATIC_FOLDER'], 'prediccion.png')
        plt.savefig(graph_path)
        plt.close()

        # Devolver la URL de la imagen generada y las predicciones
        predicciones_list = predicciones_futuras_orig.flatten().tolist()
        return jsonify({
            'message': 'Archivo procesado con éxito',
            'graph_url': graph_path,
            'predicciones': predicciones_list
        })

    except Exception as e:
        return jsonify({'error': f'Error al procesar el archivo: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
