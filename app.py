import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time

# CONFIGURACIÓN DE PÁGINA
st.set_page_config(
    page_title="Simulación de Enfriamiento",
    page_icon="☕",
    layout="wide"
)

# ESTILOS CSS
st.markdown(
    """
    <style>

    .main {
        background-color: #0f172a;
    }

    h1 {
        text-align: center;
        color: white;
        font-size: 45px;
    }

    h2, h3 {
        color: white;
    }

    .descripcion {
        text-align: center;
        color: #cbd5e1;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .temperatura-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 25px;
        font-weight: bold;
        margin-top: 10px;
        margin-bottom: 20px;
    }

    .taza {
        width: 220px;
        height: 220px;
        border-radius: 50%;
        margin: auto;
        border: 8px solid white;
        transition: 1s;
        box-shadow: 0px 0px 40px rgba(255,255,255,0.3);
    }

    .estado {
        text-align: center;
        font-size: 28px;
        color: white;
        margin-top: 20px;
        font-weight: bold;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# TÍTULO
st.markdown("""
<h1>☕ Simulación Interactiva del Enfriamiento de una Bebida</h1>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div class="descripcion">
    Modelamiento matemático usando la Ley de Enfriamiento de Newton<br>
    Métodos Numéricos: Euler y Runge-Kutta 4
    </div>
    """,
    unsafe_allow_html=True
)

# SIDEBAR
st.sidebar.header("⚙ Parámetros de Simulación")

T0 = st.sidebar.slider(
    "Temperatura Inicial (°C)",
    20,
    120,
    90
)

Ta = st.sidebar.slider(
    "Temperatura Ambiente (°C)",
    0,
    40,
    25
)

k = st.sidebar.slider(
    "Constante de Enfriamiento k",
    0.01,
    1.0,
    0.10
)

Tiempo_total = st.sidebar.slider(
    "Tiempo Total (min)",
    10,
    120,
    60
)

h = 1

iniciar = st.sidebar.button("▶ Iniciar Simulación")

# ECUACIÓN DIFERENCIAL
def f(T):
    return -k * (T - Ta)

# MÉTODO DE EULER
def euler():

    tiempos = np.arange(0, Tiempo_total + h, h)

    temperaturas = [T0]

    for i in range(len(tiempos) - 1):

        T_actual = temperaturas[-1]

        T_nueva = T_actual + h * f(T_actual)

        temperaturas.append(T_nueva)

    return tiempos, temperaturas

# RUNGE-KUTTA 4
def rk4():

    tiempos = np.arange(0, Tiempo_total + h, h)

    temperaturas = [T0]

    for i in range(len(tiempos) - 1):

        T_actual = temperaturas[-1]

        k1 = f(T_actual)
        k2 = f(T_actual + (h/2) * k1)
        k3 = f(T_actual + (h/2) * k2)
        k4 = f(T_actual + h * k3)

        T_nueva = T_actual + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

        temperaturas.append(T_nueva)

    return tiempos, temperaturas

# AJUSTE DE CURVAS
def modelo_enfriamiento(t, k_ajuste):
    return Ta + (T0 - Ta) * np.exp(-k_ajuste * t)

# FUNCIÓN COLOR SEGÚN TEMPERATURA
def obtener_color(temp):

    if temp >= 80:
        return "#ff0000"

    elif temp >= 60:
        return "#ff5c00"

    elif temp >= 40:
        return "#ffd000"

    elif temp >= 30:
        return "#00bfff"

    else:
        return "#0000ff"

# ESTADO DE LA BEBIDA
def estado_bebida(temp):

    if temp >= 80:
        return "🔥 Muy Caliente"

    elif temp >= 60:
        return "☕ Caliente"

    elif temp >= 40:
        return "🙂 Tibia"

    else:
        return "❄ Fría"

# INICIAR SIMULACIÓN
if iniciar:

    tiempos_euler, temp_euler = euler()
    tiempos_rk4, temp_rk4 = rk4()

    parametros, _ = curve_fit(
        modelo_enfriamiento,
        tiempos_rk4,
        temp_rk4
    )

    k_estimado = parametros[0]

    temp_ajustada = modelo_enfriamiento(
        tiempos_rk4,
        k_estimado
    )

    # COLUMNAS
    col1, col2 = st.columns([1, 2])

    taza_placeholder = col1.empty()
    estado_placeholder = col1.empty()
    temp_placeholder = col1.empty()
    tiempo_placeholder = col1.empty()

    grafica_placeholder = col2.empty()

    # SIMULACIÓN EN TIEMPO REAL
    for i in range(len(tiempos_rk4)):

        temp_actual = temp_rk4[i]

        color = obtener_color(temp_actual)

        estado = estado_bebida(temp_actual)

        # TAZA
        taza_placeholder.markdown(
            f'''
            <div class="taza" style="background:{color};"></div>
            ''',
            unsafe_allow_html=True
        )

        # ESTADO
        estado_placeholder.markdown(
            f'''
            <div class="estado">{estado}</div>
            ''',
            unsafe_allow_html=True
        )

        # TEMPERATURA ACTUAL
        temp_placeholder.markdown(
            f'''
            <div class="temperatura-box" style="background:{color};">
            Temperatura Actual<br>
            {temp_actual:.2f} °C
            </div>
            ''',
            unsafe_allow_html=True
        )

        # TIEMPO
        tiempo_placeholder.metric(
            "⏱ Tiempo Transcurrido",
            f"{tiempos_rk4[i]} min"
        )

        # GRÁFICA DINÁMICA
        fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(
            tiempos_euler[:i+1],
            temp_euler[:i+1],
            label='Euler',
            marker='o'
        )

        ax.plot(
            tiempos_rk4[:i+1],
            temp_rk4[:i+1],
            label='RK4',
            marker='s'
        )

        ax.plot(
            tiempos_rk4[:i+1],
            temp_ajustada[:i+1],
            '--',
            label='Ajuste de Curva'
        )

        ax.set_title('Enfriamiento de la Bebida')

        ax.set_xlabel('Tiempo (min)')

        ax.set_ylabel('Temperatura (°C)')

        ax.grid(True)

        ax.legend()

        grafica_placeholder.pyplot(fig)

        time.sleep(0.2)

    # TABLA FINAL
    st.subheader("📊 Tabla de Resultados")

    df = pd.DataFrame({
        'Tiempo': tiempos_rk4,
        'Euler': np.round(temp_euler, 2),
        'RK4': np.round(temp_rk4, 2),
        'Ajuste': np.round(temp_ajustada, 2)
    })

    st.dataframe(df)

    # RESULTADOS FINALES
    st.success(
        f"Constante k estimada por ajuste de curvas: {k_estimado:.4f}"
    )

else:

    st.info("Configure los parámetros y presione 'Iniciar Simulación'.")
