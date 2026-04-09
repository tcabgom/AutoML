Sistema AutoML desarrollado como Trabajo Fin de Máster, centrado en el diseño e implementación de un pipeline automatizado de Machine Learning.

Este proyecto experimenta con las distintas fases del ciclo de vida del aprendizaje automático (preprocesado, selección de modelos y optimización de hiperparámetros) dentro de una arquitectura modular y extensible.

## Características principales

 - Pipeline completo de AutoML (Preprocesado incluido)
 - Arquitectura modular desacoplada
 - Soporte para múltiples algoritmos:
 - Optimización de hiperparámetros:
 - Ensamble de modelos (Stacking)
 - Meta-learning
 - Soporte para paralelización (n_jobs)
 -  Control completo mediante configuración

## Instalación

```
git clone https://github.com/tcabgom/AutoML.git
cd tu-repo
pip install -r requirements.txt
```

## Configuración

El comportamiento del sistema se controla mediante `AutoMLConfig`.
