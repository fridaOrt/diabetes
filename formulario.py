import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC

# Cargar el conjunto de datos (ajusta la ruta a tu archivo CSV)
df_all = pd.read_csv('diabetes1.csv')

# Ejemplo de cómo podría verse df_all, descomenta y ajusta según sea necesario
# df_all = pd.DataFrame({
#     'Pregnancies': [0, 1, 2, 3],
#     'Glucose': [85, 89, 78, 120],
#     'BloodPressure': [66, 70, 80, 90],
#     'SkinThickness': [29, 25, 30, 35],
#     'Insulin': [0, 0, 0, 0],
#     'BMI': [26.6, 30.1, 25.8, 32.0],
#     'DiabetesPedigreeFunction': [0.351, 0.167, 0.548, 0.245],
#     'Age': [23, 25, 30, 35],
#     'Outcome': [0, 0, 1, 1]
# })

# Supongamos que df_all ya está definido con tus datos
X = df_all.iloc[:, :-1]
y = df_all.iloc[:, -1]

# División de los datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Entrenar el modelo SVM
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Título de la aplicación
st.title("Formulario de Datos de Diabetes")

# Crear un formulario
with st.form(key='diabetes_form'):
    pregnancies = st.number_input("Número de Embarazos", min_value=0, max_value=20)
    glucose = st.number_input("Glucosa", min_value=0, max_value=300)
    blood_pressure = st.number_input("Presión Arterial", min_value=0, max_value=200)
    skin_thickness = st.number_input("Grosor de la Piel", min_value=0, max_value=100)
    insulin = st.number_input("Insulina", min_value=0, max_value=500)
    bmi = st.number_input("Índice de Masa Corporal (BMI)", min_value=0.0, max_value=50.0, format="%.2f")
    diabetes_pedigree = st.number_input("Función de Pedigrí de Diabetes", min_value=0.0, max_value=2.5, format="%.2f")
    age = st.number_input("Edad", min_value=0, max_value=120)

    # Botón para enviar el formulario
    submit_button = st.form_submit_button("Enviar")

# Procesar los datos si se envían
if submit_button:
    # Crear un array con los datos ingresados
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    
    # Realizar la predicción
    y_pred_svm = clf.predict(input_data)
    
    # Mostrar los datos recibidos
    #st.success("Datos recibidos:")
    #st.write(f"Número de Embarazos: {pregnancies}")
    #st.write(f"Glucosa: {glucose}")
    #st.write(f"Presión Arterial: {blood_pressure}")
    #st.write(f"Grosor de la Piel: {skin_thickness}")
    #st.write(f"Insulina: {insulin}")
    #st.write(f"Índice de Masa Corporal (BMI): {bmi}")
    #st.write(f"Función de Pedigrí de Diabetes: {diabetes_pedigree}")
    #st.write(f"Edad: {age}")
    
    # Mostrar el resultado de la predicción
    st.success(f"Predicción de Diabetes: {'Positivo' if y_pred_svm[0] == 1 else 'Negativo'}")
    
    # Mostrar la precisión del modelo (opcional)
    accuracy = clf.score(X_test, y_test)
    st.write("Precisión del modelo en los datos de prueba:", accuracy)
    #st.write("Informe de clasificación:\n", classification_report(y_test, clf.predict(X_test)))