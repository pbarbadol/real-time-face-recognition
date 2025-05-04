# Reconocimiento facial en tiempo real con liveness y XAI

![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue)
![Status](https://img.shields.io/badge/Status-Completo-brightgreen)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Un proyecto de portfolio que implementa un sistema de reconocimiento facial en tiempo real utilizando Python, OpenCV y `face_recognition`, incorporando características avanzadas como tracking de objetos, detección básica de vida (anti-spoofing) y elementos de explicabilidad (XAI).

## Demostración en Vivo

[![Demostración del Reconocimiento Facial en Tiempo Real](https://img.youtube.com/vi/5EQtyfF96YI/hqdefault.jpg)](https://youtu.be/5EQtyfF96YI?si=xlvxFV462mBWhDn6 "Haz clic para ver la demo en YouTube")

*Haz clic en la imagen para ver una demostración en video del sistema en acción en YouTube.*

## Características Principales

*   **Reconocimiento Facial en Tiempo Real:** Identifica caras conocidas desde un stream de video (webcam) comparándolas con una base de datos de encodings precalculados.
*   **Tracking Facial Eficiente:** Utiliza trackers de OpenCV (CSRT por defecto) para seguir caras entre detecciones completas, optimizando el rendimiento. Detecta caras solo cada N frames (`DETECTION_INTERVAL`).
*   **Detección de Vida (Liveness):** Implementa una comprobación básica anti-spoofing basada en el parpadeo (calculando el Eye Aspect Ratio - EAR) para verificar si la cara detectada pertenece a una persona real presente. Marca las caras como "Vivo", "Verificando" o "Sospechoso".
*   **Manejo de Incertidumbre:** Utiliza un umbral de distancia (`CONFIDENCE_THRESHOLD`) para clasificar reconocimientos. Muestra un porcentaje de confianza para caras conocidas.
*   **Explicabilidad (XAI) Toggleable:**
    *   Visualiza los landmarks faciales (puntos clave de la cara) detectados.
    *   Para caras "Desconocido", muestra información sobre la cara conocida más cercana en la base de datos y su distancia.
    *   Se puede activar/desactivar presionando la tecla 'e'.
*   **Visualización Mejorada:** Cajas delimitadoras con padding, grosor de línea fino y colores distintivos para diferentes estados (conocido, desconocido, vivo, sospechoso).

## Tech Stack

*   **Python 3.9+**
*   **OpenCV (`opencv-contrib-python`)**: Para captura de video, procesamiento de imágenes, dibujo y algoritmos de tracking (CSRT/KCF). Es crucial instalar la versión `contrib` para los trackers.
*   **`face_recognition`**: Librería de alto nivel (basada en `dlib`) para detección de caras, cálculo de landmarks y generación de embeddings faciales de 128-d.
*   **`dlib`**: La librería subyacente para `face_recognition`. Puede requerir `cmake` y C++ Build Tools para su instalación.
*   **NumPy**: Para operaciones numéricas eficientes con arrays (imágenes, embeddings).
*   **SciPy**: Utilizado para el cálculo de distancias euclidianas en la función EAR.

## Instalación

1.  **Prerrequisitos:**
    *   Tener Anaconda o Miniconda instalado.
    *   (Windows) Puede ser necesario instalar C++ Build Tools desde Visual Studio Installer si la instalación de `dlib` falla.
    *   `cmake` (puede instalarse vía pip o conda: `pip install cmake` o `conda install cmake`).

2.  **Clonar el Repositorio:**
    ```bash
    git clone https://github.com/tu_usuario/tu_repositorio.git
    cd tu_repositorio
    ```

3.  **Crear y Activar Entorno Conda:**
    ```bash
    conda create --name face_rec_env python=3.9 -y
    conda activate face_rec_env
    ```

4.  **Instalar Dependencias:**
    *   **Opción A (Recomendado: Usando `requirements.txt`):**
        *   Crea un archivo `requirements.txt` con el siguiente contenido:
            ```txt
            numpy
            scipy
            cmake
            opencv-contrib-python
            face_recognition
            ```
        *   Instala desde el archivo:
            ```bash
            pip install -r requirements.txt
            ```
    *   **Opción B (Manual):**
        ```bash
        pip install numpy scipy cmake
        # ¡Importante instalar la versión contrib de OpenCV!
        pip install opencv-contrib-python
        # Dlib puede tardar y requerir build tools. Si falla, prueba desde conda-forge:
        # conda install -c conda-forge dlib
        pip install face_recognition
        ```
        *(Nota: `face_recognition` intentará instalar `dlib` si no lo encuentra)*

## Preparación del Dataset

1.  Crea una carpeta llamada `dataset` en la raíz del proyecto.
2.  Dentro de `dataset`, crea una subcarpeta para cada persona que quieras reconocer. El nombre de la subcarpeta será el nombre que se mostrará.
    *   Ejemplo: `dataset/Marie_Curie/`, `dataset/Tu_Nombre/`
3.  Coloca varias imágenes (.jpg, .png) de cada persona dentro de su respectiva carpeta.
    *   **Consejos:** Usa 5-15 fotos por persona, con buena calidad, diferentes ángulos (pero frontales preferiblemente) y asegúrate de que solo aparezca la cara de esa persona en cada imagen de entrenamiento.

## Uso

1.  **Codificar Caras Conocidas:**
    *   Abre tu terminal (con el entorno `face_rec_env` activado).
    *   Ejecuta el script para procesar las imágenes del `dataset` y crear el archivo de encodings:
        ```bash
        python encode_faces.py
        ```
    *   Esto generará un archivo `encodings.pickle`. Debes re-ejecutar este script cada vez que añadas/cambies imágenes en el `dataset`.

2.  **Ejecutar Reconocimiento en Tiempo Real:**
    *   Ejecuta el script principal:
        ```bash
        python real_time_recognition_toggle_xai.py
        ```
    *   Se abrirá una ventana mostrando el video de tu webcam.
    *   Las caras detectadas se mostrarán con una caja delimitadora e información.
    *   **Teclas Interactivas:**
        *   `q`: Salir de la aplicación.
        *   `e`: Activar/Desactivar las funciones de explicabilidad (landmarks y detalles de desconocidos).

## ¿Cómo Funciona? (Pipeline)

1.  **Captura y Preprocesamiento:** Se lee un frame de la webcam y se redimensiona (`FRAME_SCALE`) para acelerar el procesamiento.
2.  **Detección / Tracking:**
    *   Cada `DETECTION_INTERVAL` frames:
        *   Se detectan caras en el frame redimensionado (`face_recognition.face_locations`).
        *   Se calculan los embeddings de 128-d (`face_recognition.face_encodings`).
        *   Se calculan los landmarks faciales (`face_recognition.face_landmarks`).
        *   Se comparan los embeddings con los conocidos (`face_recognition.face_distance`).
        *   Se aplica el umbral de confianza (`CONFIDENCE_THRESHOLD`) para decidir el nombre ("Desconocido" o nombre conocido).
        *   Se intenta asociar las caras detectadas con las existentes trackeadas (usando IoU).
        *   Se (re)inicializan los trackers (`cv2.TrackerCSRT_create`) para las caras activas.
    *   En los frames intermedios:
        *   Se actualiza la posición de las cajas usando los trackers (`tracker.update`).
3.  **Liveness (EAR):**
    *   Usando los landmarks de los ojos, se calcula el Eye Aspect Ratio (EAR).
    *   Si el EAR cae por debajo de `EAR_THRESHOLD` durante `EAR_CONSEC_FRAMES_DETECT`, se cuenta un parpadeo.
    *   Se necesitan `LIVENESS_CHECKS_NEEDED` parpadeos para marcar la cara como "Vivo". Si no parpadea durante mucho tiempo, se marca como "Sospechoso".
4.  **Visualización:**
    *   Se dibujan las cajas delimitadoras (con padding) y el texto informativo (ID, Nombre, Confianza, Estado Liveness).
    *   Si la explicabilidad está activada (`e`), se dibujan los landmarks y se muestra información adicional para caras desconocidas.

## Configuración

Puedes ajustar varios parámetros al inicio del script `real_time_recognition_toggle_xai.py`:

*   `ENCODINGS_PATH`: Ruta al archivo de encodings.
*   `DETECTION_MODEL`: Modelo de detección (`hog` o `cnn`).
*   `DETECTION_INTERVAL`: Frecuencia de detección de caras.
*   `CONFIDENCE_THRESHOLD`: Umbral de distancia para reconocimiento.
*   `FRAME_SCALE`: Factor de escalado para procesamiento.
*   `TRACKER_TYPE`: Tipo de tracker de OpenCV (`CSRT` o `KCF`).
*   Parámetros de EAR (`EAR_THRESHOLD`, `EAR_CONSEC_FRAMES_DETECT`, etc.).
*   Parámetros visuales (`BOX_THICKNESS`, `BOX_PADDING`, colores).

## Limitaciones

*   **Liveness Básico:** La detección basada en EAR es simple y puede ser engañada (ej., con videos o si la persona no parpadea normalmente). No detecta ataques con máscaras o fotos de alta calidad.
*   **Sensibilidad:** El rendimiento depende de la iluminación, ángulo de la cara, oclusiones (gafas de sol, mascarillas) y calidad de las imágenes del dataset.
*   **Rendimiento:** El tracking ayuda, pero procesar múltiples caras o usar `cnn` puede ser lento sin una GPU potente.
*   **Escalabilidad:** La comparación lineal con los encodings conocidos no escala bien a miles de identidades (se necesitaría ANN como Faiss/Annoy).

## Posibles Mejoras Futuras

*   Implementar métricas de evaluación (FAR/FRR).
*   Integrar modelos de embedding SOTA (ArcFace, Facenet).
*   Añadir métodos de liveness detection más robustos.
*   Empaquetar la aplicación con Docker.
*   Crear una API REST para inferencia.

## Licencia

Distribuido bajo la Licencia MIT. Ver `LICENSE` para más información.

---

*Este proyecto fue desarrollado como parte de mi portfolio en Ingeniería de IA.*