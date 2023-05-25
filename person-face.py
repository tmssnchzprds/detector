import cv2
import numpy as np

# Cargar los archivos de pesos y configuración de YOLO para detectar personas
person_net = cv2.dnn.readNet("model/yolov3.weights", "model/yolov3.cfg")
person_classes = []
with open("model/coco.names", "r") as f:
    person_classes = [line.strip() for line in f.readlines()]

# Cargar los archivos de pesos y configuración de YOLO para detectar caras
face_net = cv2.dnn.readNet("model/face.weights", "model/face.cfg")
face_classes = []
with open("model/face.names", "r") as f:
    face_classes = [line.strip() for line in f.readlines()]

# Cargar el modelo pre-entrenado de detección de rostros
#face_cascade = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")

# Configurar la captura de video desde la cámara
cap = cv2.VideoCapture(0)
# "rtsp://AtecaAdmin:AtecaAdmin@172.17.32.126/stream1"

# Inicializar contadores
num_personas = 0
con_rostro = 0
num_total = 0

while True:
    # Leer el siguiente fotograma del video
    ret, frame = cap.read()

    # Convertir el fotograma a un formato compatible con YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Ejecutar YOLO en el fotograma para detectar personas
    person_net.setInput(blob)
    person_outs = person_net.forward(person_net.getUnconnectedOutLayersNames())

    # Analizar las salidas de YOLO para detectar personas
    person_class_ids = []
    person_confidences = []
    person_boxes = []
    for out in person_outs:
        for detection in out:
            scores = detection[5:]
            person_class_id = np.argmax(scores)
            person_confidence = scores[person_class_id]
            if person_class_id == 0 and person_confidence > 0.3:
                # Se ha detectado una persona
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                person_class_ids.append(person_class_id)
                person_confidences.append(float(person_confidence))
                person_boxes.append([x, y, w, h])
                # Detección de rostro
                # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                # if len(faces) > 0:
                    # Se ha detectado un rostro
                    # con_rostro += 1

    # Ejecutar YOLO en el fotograma para detectar caras
    face_net.setInput(blob)
    face_outs = face_net.forward(face_net.getUnconnectedOutLayersNames())

    # Analizar las salidas de YOLO para detectar caras
    face_class_ids = []
    face_confidences = []
    face_boxes = []
    for out in face_outs:
        for detection in out:
            scores = detection[5:]
            face_class_id = np.argmax(scores)
            face_confidence = scores[face_class_id]
            if face_class_id == 0 and face_confidence > 0.8:
                # Se ha detectado una cara
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                face_class_ids.append(face_class_id)
                face_confidences.append(float(face_confidence))
                face_boxes.append([x, y, w, h])

    # Dibujar las cajas delimitadoras alrededor de las personas detectadas
    person_indexes = cv2.dnn.NMSBoxes(person_boxes, person_confidences, 0.3, 0.4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(person_boxes)):
        if i in person_indexes:
            x, y, w, h = person_boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Persona", (x, y - 5), font, 0.5, (0, 255, 0), 2)

    # Dibujar las cajas delimitadoras alrededor de las personas detectadas
    face_indexes = cv2.dnn.NMSBoxes(face_boxes, face_confidences, 0.8, 0.4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(face_boxes)):
        if i in face_indexes:
            x, y, w, h = face_boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "Cara", (x, y - 5), font, 0.5, (255, 0, 0), 2)

    # Contar Personas Totales
    if num_personas > len(person_indexes):
        if con_rostro > len(face_indexes):
            num_total -= 1
        else:
            num_total += 1

    # Mostrar el número de personas detectadas y con/sin rostro en la esquina superior izquierda del video
    num_personas = len(person_indexes)
    con_rostro = len(face_indexes)
    texto1 = f"Personas Total: {num_total}"
    cv2.putText(frame, texto1, (10, frame.shape[0] - 10), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    # Mostrar el video con las cajas delimitadoras y el número de personas detectadas
    cv2.imshow("Video", frame)

    # Salir si se presiona la tecla "q"
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()