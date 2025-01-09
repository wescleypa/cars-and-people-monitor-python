#
# 
# SCRIPT DESENVOLVIDO POR WESCLEY PORTO UTILIZANDO AJUDA DA IA CHATGPT #
#
#
import cv2
import numpy as np
import winsound  # Biblioteca nativa do Windows para emitir sons
import mss  # Biblioteca para captura de tela

# Carregar o modelo YOLO pré-treinado e os arquivos de configuração
# Faça o download do arquivo yolov3.weights e yolov3.cfg do site oficial do YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Função para detectar carro
def detect_car(frame):
    CONFIDENCE_THRESHOLD = 0.4  # Limite mínimo de confiança
    NMS_THRESHOLD = 0.3         # Supressão de não-máximos

    # Obter as dimensões do quadro
    height, width, channels = frame.shape

    # Preparar a imagem para YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Variáveis para a detecção de carro
    class_ids = []
    confidences = []
    boxes = []

    # Percorrer todas as detecções
    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONFIDENCE_THRESHOLD:  # Limite de confiança para detecção
                # Obter as coordenadas da caixa delimitadora
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w // 2
                y = center_y - h // 2

                # Armazenar as informações
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aplicar a supressão de não-máximos para eliminar caixas redundantes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    # Verificar se algum carro foi detectado
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            if class_ids[i] in [0, 1, 2, 3, 7]:  # Classes relevantes
                color = (0, 255, 0)  # Verde para destacar
                label = f"{class_ids[i]}: {int(confidences[i] * 100)}%"
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                return int(confidences[i] * 100)

    return False

# Configurar a captura de tela com o mss
with mss.mss() as sct:
    monitor = sct.monitors[1]  # Monitor primário

    while True:
        # Capturar uma captura de tela
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Detectar objeto na captura de tela
        detected_object = detect_car(frame)

        if detected_object and detect_car > 50:  # Limite de confiança
            print(f"{detected_object} detectado!")
            winsound.Beep(1000, 1000)  # Emitir som de alerta
            cv2.imshow("Detecção", frame)  # Exibir a captura processada

            # Aguardar antes de fechar a janela para a próxima captura
            key = cv2.waitKey(1000)  # Aguardar 1 segundo
            if key & 0xFF == ord('q'):
                break
        else:
            # Fechar a janela caso não detecte nada e ela esteja aberta
            
            if cv2.getWindowProperty("Detecção", cv2.WND_PROP_VISIBLE) >= 1:
            
                cv2.destroyWindow("Detecção")


        # Sair se pressionar 'q' em qualquer momento
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()  # Garantir que as janelas sejam fechadas ao encerrar

