import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

st.title("📊 Asistente de Inversiones Inteligente")

# 🎯 Selección de acciones
acciones = st.multiselect("Selecciona acciones:", ["AAPL", "TSLA", "MSFT", "GOOGL"], ["AAPL"])

# 🧭 Parámetros personalizables
umbral = st.slider("Umbral de alerta (%)", 1, 10, 2)
dias_pred = st.slider("Días a predecir", 1, 15, 5)

# 🔍 Análisis y predicción por acción
for ticker in acciones:
    st.subheader(f"📈 Análisis para {ticker}")
    datos = yf.download(ticker, period="3mo", interval="1d")
    precios = datos["Close"].dropna()

    # 📊 Modelo de regresión lineal
    X = np.arange(len(precios)).reshape(-1, 1)
    y = precios.values
    modelo = LinearRegression().fit(X, y)

    # 🔮 Predicción
    futuros = np.arange(len(precios), len(precios) + dias_pred).reshape(-1, 1)
    predicciones = modelo.predict(futuros)
    cambio = (predicciones[-1] - precios.iloc[-1]) / precios.iloc[-1]

    # 🚨 Alerta de tendencia
    if abs(cambio) > umbral / 100:
        tendencia = "⬆️ subida" if cambio > 0 else "⬇️ bajada"
        st.warning(f"{tendencia} del {cambio*100:.2f}% estimada en {ticker} en {dias_pred} días.")
    else:
        st.success("✅ Sin cambios significativos previstos.")

    # 📉 Gráficas de precios y predicción
    st.line_chart(precios)
    st.line_chart(predicciones)