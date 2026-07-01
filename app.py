import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

# Configuración de la página web
st.set_page_config(page_title="Nowcasting Inflación Argentina", layout="wide")

st.title("📊 Dashboard de Nowcasting de Inflación")
st.markdown("### Modelo Econométrico Purista por Mínimos Cuadrados Ordinarios (MCO)")
st.write("Desarrollado para la materia de Economía Computacional.")

# 1. Carga y Procesamiento de Datos en Vivo
@st.cache_data # Para que la web cargue rápido y no procese el Excel en cada click
def cargar_y_procesar_datos():
    # Lee el archivo (debe estar en la misma carpeta del GitHub)
    df = pd.read_excel("datos_inflacion.xlsx")
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    # Transformaciones de variables
    df['CCL_var'] = df['CCL'].pct_change() * 100
    df['M2_var'] = df['M2'].pct_change() * 100
    df['BADLAR_diff'] = df['BADLAR'].diff()
    df['IPC_rezagado'] = df['IPC'].shift(1)
    
    # Dataset purista final
    columnas_modelo = ['Fecha', 'IPC', 'IPC_rezagado', 'M2_var', 'BADLAR_diff', 'RIPTE']
    df_modelo = df[columnas_modelo].dropna().copy()
    return df_modelo

try:
    df_modelo = cargar_y_procesar_datos()
    
    # 2. Ajuste del Modelo OLS en vivo
    Y = df_modelo['IPC']
    X = sm.add_constant(df_modelo[['IPC_rezagado', 'M2_var', 'BADLAR_diff', 'RIPTE']])
    modelo_ols = sm.OLS(Y, X).fit()

    # 3. Panel Lateral: Configuración de Escenarios (Inputs del Usuario)
    st.sidebar.header("🎛️ Configuración del Escenario")
    st.sidebar.write("Ajustá los valores para simular la inflación del mes a proyectar:")
    
    # Valores inicializados con los datos reales de Mayo que usaste
    input_ipc_rez = st.sidebar.number_input("Inflación Mes Anterior (IPC Rezagado %)", value=2.1, step=0.1)
    input_m2_var = st.sidebar.number_input("Variación de M2 (%)", value=3.38, step=0.1)
    input_badlar_diff = st.sidebar.number_input("Diferencia Tasa BADLAR (Puntos Porcentuales)", value=-6.0, step=0.5)
    input_ripte = st.sidebar.number_input("Variación de Salarios (RIPTE %)", value=2.6, step=0.1)

    # 4. Cálculo de la Predicción Dinámica
    datos_proyeccion = {
        'const': 1.0,
        'IPC_rezagado': input_ipc_rez,
        'M2_var': input_m2_var,
        'BADLAR_diff': input_badlar_diff,
        'RIPTE': input_ripte
    }
    X_proyeccion = pd.DataFrame([datos_proyeccion])
    prediccion_res = modelo_ols.get_prediction(X_proyeccion)
    summary_pred = prediccion_res.summary_frame(alpha=0.05)
    
    valor_central = summary_pred['mean'].values[0]
    limite_inf = summary_pred['obs_ci_lower'].values[0]
    limite_sup = summary_pred['obs_ci_upper'].values[0]

    # 5. Mostrar Métricas Principales en la Pantalla
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="🎯 Inflación Central Proyectada", value=f"{valor_central:.2f} %")
    with col2:
        st.metric(label="🔒 Intervalo de Confianza (95%)", value=f"[{limite_inf:.2f}% a {limite_sup:.2f}%]")

    # 6. Generación del Gráfico Dinámico
    df_modelo['IPC_predicho'] = modelo_ols.predict(X)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.set_theme(style="whitegrid")
    
    # Series históricas
    ax.plot(df_modelo['Fecha'], df_modelo['IPC'], label='Inflación Real (Indec)', color='black', linewidth=2)
    ax.plot(df_modelo['Fecha'], df_modelo['IPC_predicho'], label='Predicción del Modelo', color='red', linestyle='--', linewidth=1.5)
    
    # Punto proyectado dinámico
    fecha_proyeccion = df_modelo['Fecha'].max() + pd.DateOffset(months=1)
    ax.scatter(fecha_proyeccion, valor_central, color='red', s=150, zorder=5, label=f'Proyección: {valor_central:.2f}%')
    
    # Estética de ejes
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    ax.set_title('Histórico de Inflación vs. Ajuste del Modelo y Proyección', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variación Mensual (%)')
    ax.legend(loc='upper left')
    
    st.pyplot(fig)

    # 7. Sección para el Profesor: Diagnóstico Estadístico OLS
    st.markdown("---")
    with st.expander("🔍 Ver Reporte de Regresión OLS (Detalles para el Profesor)"):
        st.write("Estadísticos de validación globales y significatividad individual:")
        st.text(modelo_ols.summary().as_text())

except Exception as e:
    st.error(f"Error al procesar el archivo o el modelo: {e}")
    st.info("Asegurate de que el archivo 'datos_inflacion.xlsx' esté subido en la raíz de tu repositorio de GitHub.")
