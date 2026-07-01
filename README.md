# Predictor Econométrico de Inflación - Argentina

Este repositorio contiene un modelo de **Nowcasting de Inflación** desarrollado en Python utilizando Mínimos Cuadrados Ordinarios (MCO/OLS) a través de `statsmodels`.

### Variables del Modelo Purista:
* **Inercia:** IPC Rezagado (t-1)
* **Monetaria:** Variación de M2
* **Política Monetaria:** Primera diferencia de la tasa BADLAR
* **Costos/Demanda:** Variación salarial (RIPTE)

El proyecto incluye una aplicación web interactiva desplegada en Streamlit.
