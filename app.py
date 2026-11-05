Python
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import os
from audio_recorder_streamlit import audio_recorder
from datetime import datetime

# Configuración de la interfaz
st.set_page_config(page_title="Proyecto Voz Koguian", layout="centered")
st.title("🎙️ Recolección y Análisis de Voz Koguian")
st.write("Herramienta pedagógica para el registro fonético y visualización espectrográfica.")

# 1. Grabación de audio
st.subheader("1. Grabación de la palabra")
st.caption("Presiona el ícono para grabar desde el celular o computador.")
audio_bytes = audio_recorder(
    text="Toca para grabar",
    recording_color="#e8b923",
    neutral_color="#6aa36f"
)

if audio_bytes:
    # Guardar audio temporalmente
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_dir = "audios_grabados"
    os.makedirs(audio_dir, exist_ok=True)
    audio_filename = os.path.join(audio_dir, f"audio_{timestamp}.wav")
    
    with open(audio_filename, "wb") as f:
        f.write(audio_bytes)
    
    st.audio(audio_bytes, format="audio/wav")

    # 2. Generación del Espectrograma
    st.subheader("2. Análisis Acústico (Espectrograma)")
    y, sr = librosa.load(audio_filename, sr=None)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, y_axis='linear', x_axis='time', sr=sr, ax=ax, cmap='magma')
    ax.set(title='Espectrograma Acústico')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    st.pyplot(fig)
    
    st.info("Pistas de lectura: Las interrupciones verticales secas indican cortes glotales ('). Zonas de alta energía superior reflejan fricción (sh, ts).")

    # 3. Formulario de transcripción y guardado
    st.subheader("3. Registro en Base de Datos")
    palabra_escrita = st.text_input("Palabra en Koguian (aplicando acuerdos: ʉ, ts, ~, x, '):")
    significado = st.text_input("Significado en Español:")
    categoria = st.selectbox("Grupo Semántico:", [
        "Partes del cuerpo",
        "Fauna y flora",
        "Entorno / Naturaleza",
        "Familia / Sociedad",
        "Alimentos y objetos",
        "Otro"
    ])
    
    if st.button("Guardar Registro"):
        if palabra_escrita and significado:
            registro = {
                "Fecha": [timestamp],
                "Archivo_Audio": [audio_filename],
                "Palabra_Koguian": [palabra_escrita],
                "Significado": [significado],
                "Categoria": [categoria]
            }
            df = pd.DataFrame(registro)
            csv_path = "base_datos_koguian.csv"
            
            if not os.path.exists(csv_path):
                df.to_csv(csv_path, index=False)
            else:
                df.to_csv(csv_path, mode='a', header=False, index=False)
                
            st.success(f"Guardado exitoso: '{palabra_escrita}' añadida a la base de datos.")
        else:
            st.warning("Completa la palabra en Koguian y su significado antes de guardar.")
