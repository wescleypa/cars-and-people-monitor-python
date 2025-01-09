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
            if confidence > 0.5:  # Limite de confiança para detecção
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
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Verificar se algum carro foi detectado
    for i in range(len(boxes)):
        if i in indexes:
            if class_ids[i] == 0:  # ID de classe para "pessoa" YOLO
                return "pessoa"
            elif class_ids[i] == 1: # BICICLETA
                return "bicicleta"
            elif class_ids[i] == 2: # CARRO
                return "carro"
            elif class_ids[i] == 3: # MOTO
                return "moto"
            elif class_ids[i] == 7: # CAMINHÃO
                return "caminhão"
    return False

# Configurar a captura de tela com o mss
with mss.mss() as sct:
    monitor = sct.monitors[1]  # Monitor primário, você pode ajustar conforme o seu caso

    while True:
        # Capturar uma captura de tela
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)  # Converter a captura para um array do OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Converter BGRA para BGR

        # Detectar se um carro está presente no quadro
        if detect_car(frame):
            print("%s detectado!" % detect_car(frame))
            winsound.Beep(1000, 1000)  # Emitir um som de alerta (1000 Hz por 1 segundo)

        # Mostrar o quadro (opcional, mas pode ser útil para depuração)
        # cv2.imshow("Screen Capture", frame)

        # Sair do loop se pressionar 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Não se esqueça de liberar recursos ao finalizar
cv2.destroyAllWindows()
