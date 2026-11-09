import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Configuración inicial de la página en modo ancho (Wide)
st.set_page_config(page_title="Predicción de Dengue Pediátrico", page_icon="🦠", layout="wide")

# Cargar el archivo CSS personalizado
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# Título Principal
st.title("🩺 Sistema de Apoyo Clínico: Predicción de Complicaciones por Dengue Pediátrico")
st.markdown("Herramienta basada en Machine Learning para la evaluación de riesgo clínico en pacientes pediátricos.")
st.markdown("---")

# Cargar y entrenar el modelo en caché
@st.cache_resource
def cargar_modelo():
    df = pd.read_csv('MASTER_CHART_F1000Research.csv', sep=None, engine='python', on_bad_lines='skip')
    df.columns = [c.strip() for c in df.columns]
    
    y = (df['GROUPS: DENGUE FEVER OR COMPLICATED DENGUE'] == 2).astype(int)
    features = ['AGE', 'GENDER', 'DURATION OF FEVER', 'HEADACHE', 'MYALGIA', 'ABDOMINAL PAIN', 
                'RASH', 'VOMITING', 'BREATHLESSNESS', 'BLEEDING', 'BP AT ADMISSION', 'ORGANOMEGALY', 
                'HB', 'PCV', 'LOWEST WBC', 'HIGHEST WBC', 'PLATELET COUNT', 'ALT', 'CREATININE']
    
    X = df[features].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col].astype(str).str.replace(',', '.'), errors='coerce')
    X.fillna(X.median(), inplace=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    return model, features, X.median(), X_test, y_test

model, features, medianas, X_test, y_test = cargar_modelo()

# SISTEMA DE PESTAÑAS PRINCIPALES (Pantalla completa sin barra lateral)
pestana_individual, pestana_lote = st.tabs(["👤 Predicción Individual", "📁 Predicción por Lote y Métricas"])

with pestana_individual:
    st.subheader("📋 Ingrese los Parámetros Clínicos del Paciente")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 📌 Datos Demográficos y Fiebre")
        age = st.slider("Edad (años)", 1, 18, 8)
        gender = st.selectbox("Género", [0, 1], format_func=lambda x: "Femenino" if x == 0 else "Masculino")
        fever_duration = st.slider("Duración de fiebre (días)", 1, 10, 3)
        
        st.markdown("##### 🤒 Síntomas Clínicos (Parte 1)")
        headache = st.selectbox("Dolor de cabeza (Cefalea)", [0, 1], format_func=lambda x: "No" if x==0 else "Sí")
        myalgia = st.selectbox("Mialgia (Dolor muscular)", [0, 1], format_func=lambda x: "No" if x==0 else "Sí")
        abdominal_pain = st.selectbox("Dolor abdominal", [0, 1], format_func=lambda x: "No" if x==0 else "Sí")
        rash = st.selectbox("Erupción (Rash)", [0, 1], format_func=lambda x: "No" if x==0 else "Sí")

    with col2:
        st.markdown("##### 🤒 Síntomas Clínicos (Parte 2)")
        vomiting = st.selectbox("Vómitos", [0, 1], format_func=lambda x: "No" if x==0 else "Sí")
        breathlessness = st.selectbox("Dificultad respiratoria", [0, 1], format_func=lambda x: "No" if x==0 else "Sí")
        bleeding = st.selectbox("Sangrado (Hemorragias)", [0, 1], format_func=lambda x: "No" if x==0 else "Sí")
        bp_admission = st.selectbox("Alteración de Presión Arterial", [0, 1], format_func=lambda x: "Normal" if x==0 else "Hipotensión/Comprometida")
        organomegaly = st.selectbox("Organomegalia", [0, 1], format_func=lambda x: "No" if x==0 else "Sí")

    with col3:
        st.markdown("##### 🧪 Exámenes de Laboratorio")
        hb = st.number_input("Hemoglobina (g/dL)", 5.0, 20.0, 12.5)
        pcv = st.number_input("Hematocrito - PCV (%)", 20.0, 60.0, 38.0)
        lowest_wbc = st.number_input("Leucocitos Mínimos (WBC)", 1000, 30000, 5000)
        highest_wbc = st.number_input("Leucocitos Máximos (WBC)", 1000, 30000, 7000)
        platelet = st.number_input("Plaquetas (count)", 5000, 500000, 150000)
        alt = st.number_input("Transaminasa ALT (U/L)", 5.0, 500.0, 30.0)
        creatinine = st.number_input("Creatinina (mg/dL)", 0.2, 5.0, 0.6)

    datos_paciente = {
        'AGE': age, 'GENDER': gender, 'DURATION OF FEVER': fever_duration, 
        'HEADACHE': headache, 'MYALGIA': myalgia, 'ABDOMINAL PAIN': abdominal_pain, 
        'RASH': rash, 'VOMITING': vomiting, 'BREATHLESSNESS': breathlessness, 
        'BLEEDING': bleeding, 'BP AT ADMISSION': bp_admission, 'ORGANOMEGALY': organomegaly, 
        'HB': hb, 'PCV': pcv, 'LOWEST WBC': lowest_wbc, 'HIGHEST WBC': highest_wbc, 
        'PLATELET COUNT': platelet, 'ALT': alt, 'CREATININE': creatinine
    }
    df_usuario = pd.DataFrame([datos_paciente])

    st.markdown("---")
    if st.button("⚡ Ejecutar Predicción Individual", type="primary", use_container_width=True):
        prediccion = model.predict(df_usuario)[0]
        probabilidad = model.predict_proba(df_usuario)[0][1]
        
        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            st.metric(label="Probabilidad de Complicación", value=f"{probabilidad*100:.1f}%")
        with col_res2:
            if prediccion == 1 or probabilidad >= 0.4:
                st.error(f"⚠️ **ALTO RIESGO DE COMPLICACIÓN POR DENGUE**\n\n*Se sugiere hospitalización y monitorización estricta del paciente pediátrico.*")
            else:
                st.success(f"✅ **BAJO RIESGO (Dengue No Complicado)**\n\n*Manejo ambulatorio recomendado bajo estrecha vigilancia de signos de alarma.*")

with pestana_lote:
    st.subheader("📁 Predicción por Lote y Evaluación de Métricas Globales")
    st.markdown("Esta sección evalúa un conjunto completo de datos para mostrar el rendimiento global del modelo, la matriz de confusión y las predicciones masivas.")

    tipo_eval = st.radio("Seleccione la fuente de datos para el lote:", ["Usar dataset de prueba del sistema", "Subir archivo CSV personalizado"])
    
    if tipo_eval == "Subir archivo CSV personalizado":
        archivo_subido = st.file_uploader("Sube tu archivo CSV con características clínicas", type=["csv"])
        if archivo_subido is not None:
            # Carga robusta agregada aquí para evitar el error de tokenización
            df_lote = pd.read_csv(archivo_subido, sep=None, engine='python', on_bad_lines='skip')
            df_lote.columns = [c.strip() for c in df_lote.columns]
        else:
            df_lote = None
    else:
        df_lote = X_test.copy()
        df_lote['Real (Etiqueta)'] = y_test.values

    if df_lote is not None:
        st.markdown("### 📊 Vista previa de los datos del lote")
        st.dataframe(df_lote.head(10), use_container_width=True)

        if st.button("🚀 Ejecutar Predicción por Lote y Calcular Métricas", type="primary"):
            # Validar que las columnas existan en el lote subido
            faltantes = [c for c in features if c not in df_lote.columns]
            if faltantes:
                st.error(f"⚠️ El archivo subido no contiene las siguientes columnas requeridas por el modelo: {faltantes}")
            else:
                X_lote = df_lote[features].copy()
                for col in X_lote.columns:
                    X_lote[col] = pd.to_numeric(X_lote[col].astype(str).str.replace(',', '.'), errors='coerce')
                X_lote.fillna(X_lote.median(), inplace=True)

                preds_lote = model.predict(X_lote)
                probs_lote = model.predict_proba(X_lote)[:, 1]

                df_resultados = df_lote.copy()
                df_resultados['Predicción'] = preds_lote
                df_resultados['Probabilidad_Complicacion'] = np.round(probs_lote * 100, 2)

                st.markdown("### 📋 Resultados de Predicciones por Lote")
                st.dataframe(df_resultados[['AGE', 'GENDER', 'PLATELET COUNT', 'Predicción', 'Probabilidad_Complicacion']].head(15), use_container_width=True)

                # Buscar columna real o etiqueta
                col_etiqueta_real = None
                for c in df_lote.columns:
                    if 'GROUP' in c.upper() or 'REAL' in c.upper():
                        col_etiqueta_real = c
                        break

                if col_etiqueta_real or 'Real (Etiqueta)' in df_lote.columns:
                    col_real_name = col_etiqueta_real if col_etiqueta_real else 'Real (Etiqueta)'
                    y_real = (df_lote[col_real_name] == 2).astype(int) if 'GROUP' in col_real_name.upper() else df_lote[col_real_name]

                    st.markdown("---")
                    st.subheader("📈 Métricas de Rendimiento del Modelo y Matriz de Confusión")
                    
                    acc = accuracy_score(y_real, preds_lote)
                    cm = confusion_matrix(y_real, preds_lote)

                    mcol1, mcol2 = st.columns(2)
                    with mcol1:
                        st.metric(label="Exactitud Global (Accuracy)", value=f"{acc*100:.2f}%")
                        st.markdown("**Matriz de Confusión:**")
                        st.write(pd.DataFrame(cm, 
                                              columns=["Pred: No Complicado", "Pred: Complicado"], 
                                              index=["Real: No Complicado", "Real: Complicado"]))
                    with mcol2:
                        st.markdown("**Reporte de Clasificación detallado:**")
                        reporte = classification_report(y_real, preds_lote, output_dict=True)
                        st.dataframe(pd.DataFrame(reporte).transpose(), use_container_width=True)