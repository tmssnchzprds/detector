import cv2
import numpy as np

# Cargar los archivos de pesos y configuración de YOLO
net = cv2.dnn.readNet("model/yolov3.weights", "model/yolov3.cfg")
classes = []
with open("model/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Configurar la captura de video desde la cámara
cap = cv2.VideoCapture("rtsp://AtecaAdmin:AtecaAdmin@172.17.32.126/stream1")
# "rtsp://AtecaAdmin:AtecaAdmin@172.17.32.126/stream1"

while True:
    # Leer el siguiente fotograma del video
    ret, frame = cap.read()

    # Convertir el fotograma a un formato compatible con YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Ejecutar YOLO en el fotograma
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Analizar las salidas de YOLO para detectar personas
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == 0 and confidence > 0.5:
                # Se ha detectado una persona
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Dibujar las cajas delimitadoras alrededor de las personas detectadas
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Persona", (x, y - 5), font, 0.5, (0, 255, 0), 2)

    # Mostrar el número de personas detectadas en la esquina superior izquierda del video
    num_people = len(indexes)
    cv2.putText(frame, "Personas: " + str(num_people), (10, 30), font, 0.8, (0, 0, 255), 2)

    # Mostrar el video con las cajas delimitadoras y el número de personas detectadas
    cv2.imshow("Video", frame)

    # Salir si se presiona la tecla "q"
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
