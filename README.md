# TP2 – MLP para clasificación de texto (20 Newsgroups)

Clasificación de 20 Newsgroups con un MLP en PyTorch, búsqueda de hiperparámetros con Optuna (con pruning) y estudio de regularización
(L1 / L2 / L1+L2, Dropout, Batch Normalization). Se compara contra el modelo del TP1 (Softmax) y contra un baseline trivial.

## Contenido
- `TP_MLP_regularizacion.ipynb`: notebook principal (se ejecuta de punta a punta).
- `requirements.txt`: dependencias.

## Cómo correrlo
1. Abrir el notebook en Google Colab y activar GPU: *Entorno de ejecución → Cambiar tipo de entorno → T4 GPU*.
2. *Ejecutar todo*. Con GPU debería tardar pocos minutos; en CPU bastante más.
3. Local: `pip install -r requirements.txt` y abrir con Jupyter o Colab.

## Estructura del notebook
1. Datos y pre-procesamiento (TF-IDF, stopwords / stem / lemma).
2. Modelo MLP, L1/L2, Dropout y BatchNorm; utilidades (estadísticas de pesos, ruido de curvas).
3. Búsqueda con Optuna (optimizadores balanceados, pruning por mediana dentro de cada optimizador).
4. Búsqueda de una red sobreajustada y criterio de elección (4.1).
5. Regularización controlada (curvas, valores absolutos de pesos, tiempos, learning rate, ruido) y optimizadores/momentos con 3 semillas.
6. Análisis de todos los trials y conclusiones.
7. Entrenamiento final, evaluación en test y comparación con Softmax y baseline.

## Velocidad
La notebook usa una versión rápida (menos trials, épocas y variantes). La tabla "Recortes para que corra rápido" del inicio del
notebook lista cada recorte y cómo volver a la versión completa.

