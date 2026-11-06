import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Configuración inicial de la página
st.set_page_config(page_title="Predicción de Dengue Pediátrico", page_icon="🦠", layout="wide")

# Cargar el archivo CSS personalizado
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# Título de la aplicación
st.title("🩺 Sistema de Apoyo Clínico: Predicción de Complicaciones por Dengue Pediátrico")
st.markdown("Herramienta basada en Machine Learning y en el estudio clínico de referencia para la evaluación de riesgo en pacientes pediátricos.")

# Cargar y entrenar el modelo en caché para optimizar rendimiento
@st.cache_resource
def cargar_modelo():
    df = pd.read_csv('MASTER_CHART_F1000Research.csv', sep=None, engine='python', on_bad_lines='skip')
    df.columns = [c.strip() for c in df.columns]
    
    y = (df['GROUPS: DENGUE FEVER OR COMPLICATED DENGUE'] == 2).astype(int)
    features = ['AGE', 'GENDER', 'DURATION OF FEVER', 'HEADACHE', 'MYALGIA', 'ABDOMINAL PAIN', 
                'RASH', 'VOMITING', 'BREATHLESSNESS', 'BLEEDING', 'BP AT ADMISSION', 'ORGANOMEGALY', 
                'HB', 'PCV', 'LOWEST WBC', 'HIGHEST WBC', 'PLATELET COUNT', 'ALT', 'CREATININE']
    
    X = df[features].copy()
    
    # Forzar conversión de todas las columnas a numéricas (limpiando textos o errores de comas/puntos)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col].astype(str).str.replace(',', '.'), errors='coerce')
    
    # Imputar nulos con la mediana de forma segura
    X.fillna(X.median(), inplace=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    return model, features, X.median()

model, features, medianas = cargar_modelo()

# Barra lateral para ingreso de datos del paciente
st.sidebar.header("📋 Parámetros Clínicos del Paciente")

def obtener_datos_usuario():
    age = st.sidebar.slider("Edad (años)", 1, 18, 8)
    gender = st.sidebar.selectbox("Género", [0, 1], format_func=lambda x: "Femenino" if x == 0 else "Masculino")
    fever_duration = st.sidebar.slider("Duración de fiebre (días)", 1, 10, 3)
    
    st.sidebar.subheader("Síntomas Clínicos")
    headache = st.sidebar.selectbox("Dolor de cabeza (Cefalea)", [0, 1], format_func=lambda x: "No" if x==0 else "Sí")
    myalgia = st.sidebar.selectbox("Mialgia (Dolor muscular)", [0, 1], format_func=lambda x: "No" if x==0 else "Sí")
    abdominal_pain = st.sidebar.selectbox("Dolor abdominal", [0, 1], format_func=lambda x: "No" if x==0 else "Sí")
    rash = st.sidebar.selectbox("Erupción (Rash)", [0, 1], format_func=lambda x: "No" if x==0 else "Sí")
    vomiting = st.sidebar.selectbox("Vómitos", [0, 1], format_func=lambda x: "No" if x==0 else "Sí")
    breathlessness = st.sidebar.selectbox("Dificultad respiratoria", [0, 1], format_func=lambda x: "No" if x==0 else "Sí")
    bleeding = st.sidebar.selectbox("Sangrado (Hemorragias)", [0, 1], format_func=lambda x: "No" if x==0 else "Sí")
    bp_admission = st.sidebar.selectbox("Alteración de Presión Arterial", [0, 1], format_func=lambda x: "Normal" if x==0 else "Hipotensión/Comprometida")
    organomegaly = st.sidebar.selectbox("Organomegalia", [0, 1], format_func=lambda x: "No" if x==0 else "Sí")
    
    st.sidebar.subheader("Laboratorios")
    hb = st.sidebar.number_input("Hemoglobina (g/dL)", 5.0, 20.0, 12.5)
    pcv = st.sidebar.number_input("Hematocrito - PCV (%)", 20.0, 60.0, 38.0)
    lowest_wbc = st.sidebar.number_input("Leucocitos Mínimos (WBC)", 1000, 30000, 5000)
    highest_wbc = st.sidebar.number_input("Leucocitos Máximos (WBC)", 1000, 30000, 7000)
    platelet = st.sidebar.number_input("Plaquetas (count)", 5000, 500000, 150000)
    alt = st.sidebar.number_input("Transaminasa ALT (U/L)", 5.0, 500.0, 30.0)
    creatinine = st.sidebar.number_input("Creatinina (mg/dL)", 0.2, 5.0, 0.6)
    
    datos = {
        'AGE': age, 'GENDER': gender, 'DURATION OF FEVER': fever_duration, 
        'HEADACHE': headache, 'MYALGIA': myalgia, 'ABDOMINAL PAIN': abdominal_pain, 
        'RASH': rash, 'VOMITING': vomiting, 'BREATHLESSNESS': breathlessness, 
        'BLEEDING': bleeding, 'BP AT ADMISSION': bp_admission, 'ORGANOMEGALY': organomegaly, 
        'HB': hb, 'PCV': pcv, 'LOWEST WBC': lowest_wbc, 'HIGHEST WBC': highest_wbc, 
        'PLATELET COUNT': platelet, 'ALT': alt, 'CREATININE': creatinine
    }
    return pd.DataFrame([datos])

df_usuario = obtener_datos_usuario()

# Cuerpo principal de la app
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Resumen de Parámetros Ingresados")
    st.dataframe(df_usuario, use_container_width=True)

with col2:
    st.subheader("⚡ Ejecutar Predicción")
    if st.button("Evaluar Riesgo Clínico", type="primary", use_container_width=True):
        prediccion = model.predict(df_usuario)[0]
        probabilidad = model.predict_proba(df_usuario)[0][1]
        
        st.markdown("---")
        if prediccion == 1 or probabilidad >= 0.4:
            st.error(f"⚠️ **ALTO RIESGO DE COMPLICACIÓN**\n\nProbabilidad estimada: **{probabilidad*100:.1f}%**\n\n*Se sugiere hospitalización y monitorización estricta.*")
        else:
            st.success(f"✅ **BAJO RIESGO (Dengue No Complicado)**\n\nProbabilidad de complicación: **{probabilidad*100:.1f}%**\n\n*Manejo ambulatorio recomendado bajo signos de alarma.*")