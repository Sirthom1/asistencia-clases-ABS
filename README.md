# Simulación de Asistencia a Clases con ABS

Este repositorio contiene el código de un modelo de simulación basada en agentes (ABS) desarrollado para estudiar la asistencia a clases universitarias en Santiago, donde esta centrada en la Universidad de Chile. El proyecto incluye la implementación en Python usando Mesa, visualizaciones interactivas con Solara y análisis de resultados.


## Estructura del proyecto
/agent.py → definición de los estudiantes-agentes

/model.py → reglas del modelo ABS

/app.py → aplicación interactiva con Solara y Plotly

/experimento.ipynb → notebook con escenarios de simulación

/requirements.txt/ → las líbrerias necesarias para ejecutar el proyecto

## Datos geográficos

Los shapefiles oficiales del DPA 2023 se deben descargar desde el sitio del IDE:  
[https://www.geoportal.cl/geoportal/catalog/download/912598ad-ac92-35f6-8045-098f214bd9c2](https://www.geoportal.cl/geoportal/catalog/download/912598ad-ac92-35f6-8045-098f214bd9c2)  


Una vez descargados, solo descomprime la carpeta y estará lista para usar. 

## Ejecución
1. Clonar el repositorio  
2. Instalar dependencias: `pip install -r requirements.txt`  
3. Ejecutar la aplicación: `solara run app.py` 

Con eso ya tienes la app interactiva para poder correr una simulación y ver sus resultados, donde mostrará una barra lateral para poder modificar los parámetros a su preferencia.

Sí quisiera hacer más de una simulación, necesitando 100 iteraciones por ejemplo, puede revisar y/o usar el archivo experimento, donde se realizarón diferentes escenarios para observar el modelo, hay puede agregar o modificar lo que hay. De la misma manera, hay funciones generales definidas para poder replicar y/o crear nuevos.
