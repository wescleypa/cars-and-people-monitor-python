import cv2
import numpy as np
import winsound  # Biblioteca nativa do Windows para emitir sons
import mss  # Biblioteca para captura de tela

# Carregar o modelo YOLO pré-treinado e os arquivos de configuração
# Faça o download do arquivo yolov3.weights e yolov3.cfg do site oficial do YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Dicionários para armazenar as informações de veículos
vehicle_ids = {}  # Dicionário para armazenar as posições dos carros com seus IDs
vehicle_counter = 0  # Contador para gerar IDs únicos
vehicle_positions_previous = {}  # Para armazenar as posições anteriores dos carros

# Função para detectar carro
def detect_car(frame, vehicle_ids, vehicle_counter, vehicle_positions_previous):
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
            if confidence > 0.5:  # Limite de confiança para detecção
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
        if i in indexes and class_ids[i] in (0,1,2,3,7):  # Verifica se é um carro (ID 2 no YOLO)
            x, y, w, h = boxes[i]
            vehicle_position = (x, y, w, h)

            vehicle_found = False
            for vehicle_id, position in vehicle_ids.items():
                prev_x, prev_y, prev_w, prev_h = position
                #print(prev_x, x)
                # Se a posição anterior do carro for próxima, tratamos como o mesmo carro
                if abs(prev_x - x) < 5 and abs(prev_y - y) < 5:
                    #print('mesma posição')
                    # Se o carro ainda está na mesma posição, não faz nada
                    vehicle_found = True
                    break

            # Caso o carro tenha se movido
            if not vehicle_found:
                print('saiu da posição')
                vehicle_ids[vehicle_counter] = vehicle_position
                detected_objects.append(vehicle_counter)
                vehicle_positions_previous[vehicle_counter] = vehicle_position  # Define a posição anterior
                vehicle_counter += 1

                # Desenhar o quadrado verde ao redor do carro
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Caixa verde
                object_name = { 0: 'pedestre', 1: 'bicicleta', 2: 'carro', 3: 'moto', 7: 'caminhao' }
                # Coloca o texto ao lado do quadrado verde
                cv2.putText(frame, object_name[class_ids[i]], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    return detected_objects, frame, vehicle_ids, vehicle_counter, vehicle_positions_previous

# Configurar a captura de tela com o mss
with mss.mss() as sct:
    monitor = sct.monitors[1]  # Monitor primário

    while True:
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        detected_objects, current_frame, vehicle_ids, vehicle_counter, vehicle_positions_previous = detect_car(
            frame, vehicle_ids, vehicle_counter, vehicle_positions_previous
        )

        if detected_objects:
            print(f"Veículos detectados: {', '.join(map(str, detected_objects))}")
            cv2.imshow("Detecção", current_frame)

        key = cv2.waitKey(1000)  # Aguardar 1 segundo
        if key & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()  # Garantir que as janelas sejam fechadas ao encerrar
