# DecisionLab — Analítica avanzada de campañas bancarias

Aplicación académica en Python y Streamlit para explorar campañas bancarias reales y estimar la probabilidad de aceptación de un depósito a plazo.

## Archivos necesarios

- `app.py`: interfaz, visualizaciones, preprocesamiento y modelo predictivo.
- `requirements.txt`: bibliotecas requeridas.
- `bank_data.zip`: datos originales (`bank-full.csv`) de UCI.

## Instalación local

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Publicación en Streamlit Community Cloud

1. Crea un repositorio en GitHub y sube estos archivos a la raíz.
2. Abre https://share.streamlit.io/ e inicia sesión con GitHub.
3. Selecciona **Create app** y el repositorio recién creado.
4. En archivo principal indica `app.py` y despliega.
5. Espera a que se instalen las dependencias y se entrene el modelo inicial.

## Nota metodológica

La regresión logística se entrena al iniciar la app (resultado almacenado en caché), usando una división estratificada 75/25 y excluyendo la duración de llamada para prevenir fuga temporal. Las probabilidades son aproximaciones estadísticas con fines educativos.

## Fuente

S. Moro, P. Rita y P. Cortez (2014). *Bank Marketing*, UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/222/bank+marketing — licencia CC BY 4.0.
