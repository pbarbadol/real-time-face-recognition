import face_recognition
import cv2
import numpy as np
import pickle
import time
from scipy.spatial import distance as dist # Para EAR

# --- Parámetros Configurables ---
ENCODINGS_PATH = "encodings.pickle"
DETECTION_MODEL = 'cnn' # 'hog' o 'cnn'
DETECTION_INTERVAL = 3 # Detectar cada N frames
CONFIDENCE_THRESHOLD = 0.55 # Distancia máxima para reconocer (más bajo = más estricto)
FRAME_SCALE = 0.50 # Escala de procesamiento (más pequeño = más rápido)
TRACKER_TYPE = "CSRT" # 'CSRT', 'KCF'

# Parámetros de Liveness (EAR)
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES_DETECT = 3
EAR_CONSEC_FRAMES_LIVE = 50
LIVENESS_CHECKS_NEEDED = 2

# Parámetros Visuales y Explicabilidad
BOX_THICKNESS = 1
BOX_PADDING = 15
KNOWN_COLOR = (150, 255, 0) # Verde lima
UNKNOWN_COLOR = (0, 165, 255) # Naranja
LIVE_COLOR = (255, 255, 0) # Cyan (para landmarks/indicador)
TEXT_COLOR = (255, 255, 255) # Blanco
LANDMARK_RADIUS = 1
show_explainability = True # Estado inicial del toggle (True para mostrar XAI)

# --- Funciones Auxiliares --- (calculate_ear, create_tracker, calculate_iou)

def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C) if C > 0 else 0.0 # Evitar division por cero
    return ear

def create_tracker(tracker_type):
    if tracker_type == 'CSRT':
        return cv2.TrackerCSRT_create()
    elif tracker_type == 'KCF':
        return cv2.TrackerKCF_create()
    else:
        print(f"[WARN] Tracker '{tracker_type}' no reconocido, usando CSRT.")
        return cv2.TrackerCSRT_create()

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB_A = boxA[0] + boxA[2]; yB_A = boxA[1] + boxA[3]
    xB_B = boxB[0] + boxB[2]; yB_B = boxB[1] + boxB[3]
    xB = min(xB_A, xB_B); yB = min(yB_A, yB_B)
    interWidth = xB - xA; interHeight = yB - yA
    if interWidth <= 0 or interHeight <= 0: interArea = 0
    else: interArea = interWidth * interHeight
    boxAArea = boxA[2] * boxA[3]; boxBArea = boxB[2] * boxB[3]
    unionArea = float(boxAArea + boxBArea - interArea)
    iou = interArea / unionArea if unionArea > 0 else 0.0
    return iou

# --- Inicialización ---

print("[INFO] Cargando encodings...")
try:
    with open(ENCODINGS_PATH, "rb") as f:
        data = pickle.load(f)
    known_encodings = data["encodings"]
    known_names = data["names"]
except FileNotFoundError:
    print(f"[ERROR] Archivo {ENCODINGS_PATH} no encontrado. Ejecuta encode_faces.py")
    exit()
except Exception as e:
    print(f"[ERROR] No se pudo cargar {ENCODINGS_PATH}: {e}")
    exit()

if not known_encodings:
    print("[ERROR] No hay encodings cargados.")
    exit()

print("[INFO] Iniciando stream de video...")
video_capture = cv2.VideoCapture(0)
time.sleep(1.0)

if not video_capture.isOpened():
    print("[ERROR] No se pudo abrir la cámara.")
    exit()

# Variables de estado
frame_count = 0
tracked_faces = {}
next_face_id = 0
latest_face_distances = [] # Guardar las distancias de la última detección para XAI

# --- Bucle Principal ---
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("[INFO] Fin del stream o error.")
        break

    height, width = frame.shape[:2]
    small_frame = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
    scale_factor = 1 / FRAME_SCALE
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    current_boxes_ids = []

    # --- Decidir si Detectar o Trackear ---
    if frame_count % DETECTION_INTERVAL == 0:
        print(f"[INFO] Frame {frame_count}: Detectando...")
        face_locations = face_recognition.face_locations(rgb_small_frame, model=DETECTION_MODEL)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame, face_locations)
        latest_face_distances = [] # Resetear en cada detección

        new_tracked_faces = {}

        for i, face_encoding in enumerate(face_encodings):
            name = "Desconocido"
            confidence = 0.0
            best_match_dist = 1.0

            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=CONFIDENCE_THRESHOLD + 0.05)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            latest_face_distances.append(face_distances) # Guardar para posible XAI

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                best_match_dist = face_distances[best_match_index]

                if matches[best_match_index] and best_match_dist < CONFIDENCE_THRESHOLD:
                    name = known_names[best_match_index]
                    confidence = (1.0 - best_match_dist) * 100

            top, right, bottom, left = face_locations[i]
            box_orig = (int(top * scale_factor), int(right * scale_factor),
                        int(bottom * scale_factor), int(left * scale_factor))

            landmarks_orig = {}
            if i < len(face_landmarks_list):
                 landmarks_orig = {key: [(int(p[0] * scale_factor), int(p[1] * scale_factor)) for p in value]
                                  for key, value in face_landmarks_list[i].items()}

            found_existing_id = -1
            best_iou = 0.0
            (xt, yt, wb, hb) = (box_orig[3], box_orig[0], box_orig[1]-box_orig[3], box_orig[2]-box_orig[0])
            for face_id, data in tracked_faces.items():
                 (xo, yo, wo, ho) = (data['box'][3], data['box'][0], data['box'][1]-data['box'][3], data['box'][2]-data['box'][0])
                 iou = calculate_iou((xt, yt, wb, hb), (xo, yo, wo, ho))
                 if iou > 0.3 and iou > best_iou:
                      best_iou = iou
                      found_existing_id = face_id

            if found_existing_id == -1:
                face_id = next_face_id
                next_face_id += 1
                tracker = create_tracker(TRACKER_TYPE)
                tracker_bbox = (box_orig[3], box_orig[0], box_orig[1] - box_orig[3], box_orig[2] - box_orig[0])
                try:
                    tracker.init(frame, tracker_bbox)
                    new_tracked_faces[face_id] = {
                        'tracker': tracker, 'name': name, 'confidence': confidence, 'box': box_orig,
                        'landmarks': landmarks_orig, 'ear_counter': 0, 'blink_counter': 0,
                        'liveness_status': "Verificando", 'frames_since_blink': 0, 'best_dist': best_match_dist,
                        'detection_index': i # Guardar el índice original para vincular con latest_face_distances
                    }
                    current_boxes_ids.append(face_id)
                except Exception as e:
                    print(f"[WARN] No se pudo inicializar tracker para nueva cara: {e}")
                    pass # Fallo silencioso si no se puede trackear
            else:
                 tracked_faces[found_existing_id]['name'] = name
                 tracked_faces[found_existing_id]['confidence'] = confidence
                 tracked_faces[found_existing_id]['box'] = box_orig
                 tracked_faces[found_existing_id]['landmarks'] = landmarks_orig
                 tracked_faces[found_existing_id]['best_dist'] = best_match_dist
                 tracked_faces[found_existing_id]['detection_index'] = i # Actualizar índice
                 tracker = create_tracker(TRACKER_TYPE)
                 tracker_bbox = (box_orig[3], box_orig[0], box_orig[1] - box_orig[3], box_orig[2] - box_orig[0])
                 try:
                      tracker.init(frame, tracker_bbox)
                      tracked_faces[found_existing_id]['tracker'] = tracker
                 except Exception as e:
                       print(f"[WARN] No se pudo re-inicializar tracker para cara {found_existing_id}: {e}") # Opcional
                       del tracked_faces[found_existing_id] # Eliminar si falla
                       continue

                 new_tracked_faces[found_existing_id] = tracked_faces[found_existing_id]
                 current_boxes_ids.append(found_existing_id)

        tracked_faces = new_tracked_faces

    else:
        # --- Fase de Tracking ---
        ids_to_remove = []
        for face_id, data in tracked_faces.items():
            success, box = data['tracker'].update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in box]
                data['box'] = (y, x + w, y + h, x) # top, right, bottom, left
                current_boxes_ids.append(face_id)
            else:
                print(f"[INFO] Tracker para '{data['name']}' (ID {face_id}) perdido.") # Opcional
                ids_to_remove.append(face_id)

        for face_id in ids_to_remove:
            del tracked_faces[face_id]

    # --- Procesamiento Post-Detección/Tracking (Liveness) ---
    faces_to_process = list(tracked_faces.keys())
    for face_id in faces_to_process:
        if face_id not in tracked_faces: continue
        data = tracked_faces[face_id]

        if data['landmarks'] and 'left_eye' in data['landmarks'] and 'right_eye' in data['landmarks']:
            left_eye = data['landmarks']['left_eye']
            right_eye = data['landmarks']['right_eye']
            left_ear = calculate_ear(left_eye); right_ear = calculate_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0

            if ear < EAR_THRESHOLD:
                data['ear_counter'] += 1
                data['frames_since_blink'] = 0
            else:
                if data['ear_counter'] >= EAR_CONSEC_FRAMES_DETECT:
                    data['blink_counter'] += 1
                    # print(f"[LIVENESS] Parpadeo detectado ID {face_id}! Total: {data['blink_counter']}") # Opcional
                    if data['liveness_status'] != "Vivo" and data['blink_counter'] >= LIVENESS_CHECKS_NEEDED:
                         data['liveness_status'] = "Vivo"
                         print(f"[LIVENESS] ID {face_id} verificado como Vivo.") # Opcional
                data['ear_counter'] = 0
                data['frames_since_blink'] += 1

            if data['liveness_status'] == "Vivo" and data['frames_since_blink'] > EAR_CONSEC_FRAMES_LIVE * 5 :
                 print(f"[LIVENESS] ID {face_id} inactivo, requiere nueva verificación.") # Opcional
                 data['liveness_status'] = "Verificando"
                 data['blink_counter'] = 0

            if data['liveness_status'] == "Verificando" and data['frames_since_blink'] > EAR_CONSEC_FRAMES_LIVE:
                  data['liveness_status'] = "Sospechoso" # Simplificado


    # --- Dibujar Resultados ---
    frame_draw = frame.copy()
    faces_to_draw = list(tracked_faces.keys())
    for face_id in faces_to_draw:
        if face_id not in tracked_faces: continue
        data = tracked_faces[face_id]
        (top, right, bottom, left) = data['box']
        name = data['name']
        confidence = data['confidence']
        liveness = data['liveness_status']
        landmarks = data['landmarks']

        pad_top = max(0, top - BOX_PADDING); pad_left = max(0, left - BOX_PADDING)
        pad_bottom = min(height, bottom + BOX_PADDING); pad_right = min(width, right + BOX_PADDING)

        display_text = f"ID:{face_id} "
        box_color = UNKNOWN_COLOR

        if name != "Desconocido":
             display_text += f"{name} ({confidence:.0f}%)"
             if liveness == "Vivo":
                 display_text += " - Vivo"
                 box_color = KNOWN_COLOR
             elif liveness == "Verificando":
                 display_text += " - Verificando..."
                 box_color = (255, 200, 0)
             else: # Sospechoso
                 display_text += " - Sospechoso"
                 box_color = (0, 0, 255)
        else:
            display_text += "Desconocido"
            # Añadir explicabilidad si está activada
            if show_explainability and 'detection_index' in data and data['detection_index'] < len(latest_face_distances):
                 distances = latest_face_distances[data['detection_index']]
                 if len(distances) > 0:
                    best_dist_unknown = np.min(distances)
                    closest_name_idx = np.argmin(distances)
                    closest_name = known_names[closest_name_idx]
                    display_text += f" (Más prox: {closest_name} Dist:{best_dist_unknown:.2f})"

            display_text += f" - {liveness}"
            box_color = UNKNOWN_COLOR if liveness != "Sospechoso" else (0, 0, 255)

        cv2.rectangle(frame_draw, (pad_left, pad_top), (pad_right, pad_bottom), box_color, BOX_THICKNESS)

        (tw, th), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_draw, (pad_left, pad_top - th - 5), (pad_left + tw, pad_top), box_color, -1)
        cv2.putText(frame_draw, display_text, (pad_left + 3, pad_top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

        # Dibujar Landmarks si la explicabilidad está activada
        if show_explainability and landmarks:
            lm_color = LIVE_COLOR if liveness == "Vivo" else box_color
            for feature, points in landmarks.items():
                 for (x, y) in points:
                      cv2.circle(frame_draw, (x, y), LANDMARK_RADIUS, lm_color, -1)


    # --- Manejo de Teclas ---
    key = cv2.waitKey(1) & 0xFF

    # Salir con 'q'
    if key == ord('q'):
        break

    # Toggle de explicabilidad con 'e'
    if key == ord('e'):
        show_explainability = not show_explainability
        status = "ACTIVADA" if show_explainability else "DESACTIVADA"
        print(f"[INFO] Explicabilidad (landmarks/distancia desconocidos) {status}")


    frame_count += 1
    cv2.imshow('Reconocimiento Facial (Q: Salir, E: Toggle XAI)', frame_draw)


# --- Limpieza Final ---
print(f"\n[INFO] Frames procesados: {frame_count}")
video_capture.release()
cv2.destroyAllWindows()
print("[INFO] Stream finalizado y recursos liberados.")