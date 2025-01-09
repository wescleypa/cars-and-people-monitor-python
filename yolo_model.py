import cv2

def load_yolo_model():
    # Carregar o modelo YOLO pré-treinado e os arquivos de configuração
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers
