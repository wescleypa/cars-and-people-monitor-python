import cv2
import mss
import time
import pygame
import numpy as np
from yolo_model import load_yolo_model
from object_detection import detect_car

# Inicializando o pygame para reproduzir sons
pygame.mixer.init()

# Carregar o som desejado para alerta
alert_sound = pygame.mixer.Sound('alert_sound.mp3')

# Carregar o modelo YOLO
net, output_layers = load_yolo_model()

# Dicionários para armazenar as informações de veículos
vehicle_ids = {}
vehicle_counter = 0
vehicle_positions_previous = {}

last_detection_time = None
detected_last = False
real_detect = None
last_detect = None

# Configurar a captura de tela com o mss
with mss.mss() as sct:
    monitor = sct.monitors[1]  # Monitor primário

    while True:
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        detected_objects, current_frame, vehicle_ids, vehicle_counter, vehicle_positions_previous = detect_car(
            frame, vehicle_ids, vehicle_counter, vehicle_positions_previous, net, output_layers
        )

        if detected_objects:
            alert_sound.play()
            print(f"Veículos detectados: {', '.join(map(str, detected_objects))}")

            # Atualizando as variáveis de detecção se o objeto foi detectado
            real_detect = vehicle_ids[detected_objects[0]]  # Pegando a posição do carro detectado

            # Verificando se o carro detectado é diferente do anterior
            if last_detect != real_detect:
                last_detect = real_detect  # Atualizando a última detecção
                last_detection_time = time.time()  # Reiniciando o tempo de contagem
                detected_last = True  # Marcando que houve uma detecção recente

            # Cortar a imagem para focar no veículo detectado
            for vehicle_id in detected_objects:
                x, y, w, h = vehicle_ids[vehicle_id]  # Pega a posição do veículo detectado
                cropped_frame = current_frame[y:y+300, x:x+300]  # Realiza o corte da imagem

                # Exibir a imagem recortada (se você quiser visualizar o foco no veículo)
                cv2.imshow("Corte do Carro", cropped_frame)

            #cv2.imshow("Detecção", current_frame)
        # Verificar se passaram 5 segundos desde a última detecção
        if detected_last and last_detection_time and (time.time() - last_detection_time >= 5):
            cv2.destroyAllWindows()  # Fechar as janelas automaticamente após 5 segundos
            print("Fechando a janela devido à inatividade.")
            last_detection_time = None  # Resetando a contagem para aguardar nova detecção
            detected_last = False  # Resetando o flag para indicar que não há mais objeto detectado

        key = cv2.waitKey(1000)  # Aguardar 1 segundo
        if key & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()  # Garantir que as janelas sejam fechadas ao encerrar