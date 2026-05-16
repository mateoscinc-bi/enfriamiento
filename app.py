import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# =============================
# CONFIGURACIÓN
# =============================
st.set_page_config(page_title="☕ Enfriamiento", layout="wide")

# =============================
# ESTILO VISUAL
# =============================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1f1c2c, #928dab);
    color: white;
}
h1, h2, h3 {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("☕ Simulación de Enfriamiento de una Bebida")
st.subheader("Ley de Enfriamiento de Newton")

# =============================
# ENTRADAS
# =============================
col1, col2, col3 = st.columns(3)

with col1:
    T0 = st.number_input("🌡️ Temp inicial (°C)", value=90.0)

with col2:
    Ta = st.number_input("🌬️ Temp ambiente (°C)", value=25.0)

with col3:
    k = st.number_input("⚙️ Constante k", value=0.1)

tiempo_total = st.slider("⏳ Tiempo (min)", 1, 60, 20)

# =============================
# MODELO
# =============================
def f(T):
    return -k * (T - Ta)

tiempos = np.arange(0, tiempo_total + 1, 1)

# Euler
T_euler = [T0]
for i in range(len(tiempos)-1):
    T_euler.append(T_euler[-1] + f(T_euler[-1]))

# RK4
T_rk4 = [T0]
for i in range(len(tiempos)-1):
    T = T_rk4[-1]
    k1 = f(T)
    k2 = f(T + 0.5*k1)
    k3 = f(T + 0.5*k2)
    k4 = f(T + k3)
    T_rk4.append(T + (k1 + 2*k2 + 2*k3 + k4)/6)

# Exacta
T_exacta = Ta + (T0 - Ta) * np.exp(-k * tiempos)

# =============================
# COLOR (ROJO → AZUL)
# =============================
def color_temp(T):
    if T0 == Ta:
        return "blue"
    ratio = (T - Ta) / (T0 - Ta)
    ratio = max(0, min(1, ratio))
    r = int(255 * ratio)
    b = int(255 * (1 - ratio))
    return (r/255, 0, b/255)

# =============================
# ANIMACIÓN
# =============================
st.subheader("🔥 Animación del Enfriamiento")

grafico_animado = st.empty()

for i in range(len(tiempos)):
    fig, ax = plt.subplots(figsize=(4,4))

    color = color_temp(T_rk4[i])

    # taza (círculo)
    taza = plt.Circle((0, 0), 1, color=color)
    ax.add_patch(taza)

    # detalles de taza
    handle = plt.Circle((1.2, 0), 0.3, fill=False, linewidth=3)
    ax.add_patch(handle)

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    ax.set_title(f"Minuto {tiempos[i]}\nTemp: {T_rk4.2f} °C")
    ax.axis('off')

    grafico_animado.pyplot(fig)
    time.sleep(0.15)

# =============================
# GRÁFICA
# =============================
st.subheader("📈 Comparación de Métodos")

fig2, ax2 = plt.subplots()

ax2.plot(tiempos, T_euler, 'o-', label="Euler")
ax2.plot(tiempos, T_rk4, 's-', label="RK4")
ax2.plot(tiempos, T_exacta, '--', label="Exacta")

ax2.set_xlabel("Tiempo (min)")
ax2.set_ylabel("Temperatura (°C)")
ax2.legend()
ax2.grid()

st.pyplot(fig2)

# =============================
# TABLA
# =============================
st.subheader("📊 Tabla de Datos")

df = pd.DataFrame({
    "Tiempo": tiempos,
    "Euler": np.round(T_euler, 2),
    "RK4": np.round(T_rk4, 2),
    "Exacta": np.round(T_exacta, 2)
})

st.dataframe(df, use_container_width=True)

# =============================
# ERRORES
# =============================
error_euler = np.max(np.abs(T_exacta - T_euler))
error_rk4 = np.max(np.abs(T_exacta - T_rk4))

st.subheader("📉 Precisión")

st.write(f"🔴 Error Euler: {error_euler:.4f}")
st.write(f"🟢 Error RK4: {error_rk4:.4f}")

# =============================
# CONCLUSIÓN
# =============================
st.subheader("🧠 Conclusión")

if error_rk4 < error_euler:
    st.success("✅ Runge-Kutta es más preciso que Euler.")
else:
    st.warning("⚠️ Revisar parámetros.")