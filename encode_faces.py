import face_recognition
import os
import cv2
import pickle # Para guardar los encodings
import time

print("[INFO] Iniciando codificación de caras...")
start_time = time.time()

# Ruta a la carpeta con las imágenes de las personas conocidas
dataset_path = 'dataset'
known_encodings = []
known_names = []

# Iterar sobre cada persona en el dataset
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue

    print(f"[+] Procesando a: {person_name}")

    # Iterar sobre cada imagen de la persona
    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        print(f"  - Procesando imagen: {image_name}")

        try:
            # Cargar la imagen
            image = face_recognition.load_image_file(image_path)
            # Convertir a RGB (face_recognition lo hace internamente, pero es buena práctica)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detectar la cara (asumimos una sola cara por imagen de entrenamiento)
            boxes = face_recognition.face_locations(image, model='cnn')

            # Calcular el embedding de la cara
            # Usar known_face_locations para asegurar que usamos la cara detectada
            encodings = face_recognition.face_encodings(image, known_face_locations=boxes)

            # Añadir el primer encoding encontrado (asumiendo una cara)
            if encodings:
                encoding = encodings[0]
                known_encodings.append(encoding)
                known_names.append(person_name)
            else:
                 print(f"  [WARN] No se detectó cara en {image_name}")

        except Exception as e:
            print(f"  [ERROR] Procesando {image_name}: {e}")

# Guardar los encodings y nombres
print("\n[INFO] Guardando encodings...")
data = {"encodings": known_encodings, "names": known_names}
output_file = "encodings.pickle"
with open(output_file, "wb") as f:
    f.write(pickle.dumps(data))

end_time = time.time()
print(f"[INFO] Codificación completada en {end_time - start_time:.2f} segundos.")
print(f"[INFO] Encodings guardados en {output_file}")