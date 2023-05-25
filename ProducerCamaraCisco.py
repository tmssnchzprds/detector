from kafka import KafkaProducer
from json import dumps
import cv2
import numpy as np
import time
import datetime
import holidays

producer = KafkaProducer(
    bootstrap_servers=['172.17.10.33:9092','172.17.10.34:9092','172.17.10.35:9092'])

# Cargar los archivos de pesos y configuración de YOLO
net = cv2.dnn.readNet("model/yolov3.weights", "model/yolov3.cfg")
classes = []
with open("model/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Configurar la captura de video desde la cámara
cap = cv2.VideoCapture("rtsp://AtecaAdmin:AtecaAdmin@172.17.32.83/stream1")
# "rtsp://AtecaAdmin:AtecaAdmin@172.17.32.126/stream1"

# Establecer el tiempo de inicio
start_time = time.time()

def es_dia_lectivo(fecha):
    es_festivo = fecha in holidays.Spain()
    es_fin_semana = fecha.weekday() >= 5  # 5 es sábado, 6 es domingo

    return not (es_festivo or es_fin_semana)

while True:
    # Leer el siguiente fotograma del video
    ret, frame = cap.read()

    # Verificar si han pasado 1 minutos
    elapsed_time = time.time() - start_time
    if elapsed_time >= 15:  # 60 segundos = 1 minutos
        
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
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #for i in range(len(boxes)):
            #if i in indexes:
                #x, y, w, h = boxes[i]
                #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                #cv2.putText(frame, "Persona", (x, y - 5), font, 0.5, (0, 255, 0), 2)
        
        # Mostrar el número de personas detectadas en la esquina superior izquierda del video
        fechaHora = datetime.datetime.now()
        anio = fechaHora.year
        mes = fechaHora.month
        dia = fechaHora.day
        hora = fechaHora.hour
        min = fechaHora.minute
        sec = fechaHora.second
        date = f"{anio}{mes:02}{dia:02}{hora:02}{min:02}{sec:02}"
        num_people = len(indexes)
        if num_people == 0:
            detection = False
        else:
            detection = True
        if es_dia_lectivo(fechaHora) and 8 < hora < 22:
            lectivo = True
        else:
            lectivo = False
        
        #cv2.putText(frame, "Personas: " + str(num_people), (10, 30), font, 0.8, (0, 0, 255), 2)
        #print(num_people)

        # Mostrar el video con las cajas delimitadoras y el número de personas detectadas
        #cv2.imshow("Video", frame)
        producer.send("camaraAULAcisco", value=dumps({"Fecha": date, "Horario Lectivo": lectivo, "Deteccion": detection, "Cantidad":  num_people}).encode('utf-8'))
        # Como el envío es asíncrono, para que no se salga del programa antes de enviar el mensaje, esperamos 1 seg
        time.sleep(1)
        # Reiniciar el tiempo de inicio
        start_time = time.time()

    # Salir si se presiona la tecla "q"
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
