import cv2
import numpy as np
import configparser

# Criar o objeto ConfigParser
config = configparser.ConfigParser()
# Ler o arquivo .cfg com encoding UTF-8
config.read("config.cfg", encoding="utf-8")

# Mapear os objetos com os IDs correspondentes
object_map = {
    "pedestre": config["geral"].getint("pedestre"),
    "carro":    config["geral"].getint("carro"),
    "moto":     config["geral"].getint("moto"),
    "bike":     config["geral"].getint("bike"),
    "caminhao": config["geral"].getint("caminhao")
}

config = config["object_detection"] # DEFINE GRUPO DE CONFIGURAÇÃO

# DEFINE OS OBJETOS ATIVOS
active_objects = [id for key, id in object_map.items() if config.getboolean(key)]

# FUNÇÃO PARA DETECTAR OBJETOS
def detect_car(frame, vehicle_ids, vehicle_counter, vehicle_positions_previous, net, output_layers):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence >  0.5:  # Limite de confiança para detecção
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w // 2
                y = center_y - h // 2

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    detected_objects = []

    for i in range(len(boxes)):
        if i in indexes and class_ids[i] in active_objects:
            x, y, w, h = boxes[i]
            vehicle_position = (x, y, w, h)

            vehicle_found = False
            for vehicle_id, position in vehicle_ids.items():
                prev_x, prev_y, prev_w, prev_h = position
                if abs(prev_x - x) < 6\
                    and abs(prev_y - y) < 6:
                    vehicle_found = True
                    break

            if not vehicle_found:
                vehicle_ids[vehicle_counter] = vehicle_position
                detected_objects.append(vehicle_counter)
                vehicle_positions_previous[vehicle_counter] = vehicle_position
                vehicle_counter += 1

                # Desenhar o quadrado verde ao redor do carro
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                object_name = {0: 'pedestre', 1: 'bicicleta', 2: 'carro', 3: 'moto', 7: 'caminhao'}
                cv2.putText(frame, object_name[class_ids[i]], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    return detected_objects, frame, vehicle_ids, vehicle_counter, vehicle_positions_previous
