import dlib
import cv2
import imutils
import matplotlib.pyplot as plt
from imutils import face_utils

# Carrega o detector de faces do dlib
detector_faces = dlib.get_frontal_face_detector()

# Carrega o modelo de pontos de referência faciais do dlib
predictor_pontos = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Carrega a imagem de entrada usando o OpenCV
imagem_entrada = cv2.imread("Data/3.jpg")

# Converte a imagem para escala de cinza
imagem_cinza = cv2.cvtColor(imagem_entrada, cv2.COLOR_BGR2GRAY)

# Detecta os rostos na imagem usando o detector de faces
rects = detector_faces(imagem_cinza)

# Define a função "landmark" para marcar os pontos de referência faciais nos rostos detectados
def landmark(rects):
    for (i, rect) in enumerate(rects):
        # Obtém os pontos de referência faciais usando o modelo do dlib
        pontos_referencia = predictor_pontos(imagem_cinza, rect)
        pontos_referencia = face_utils.shape_to_np(pontos_referencia)

        # Desenha um retângulo ao redor do rosto detectado
        #(x, y, w, h) = face_utils.rect_to_bb(rect)
        #cv2.rectangle(imagem_entrada, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Desenha círculos nos pontos de referência faciais
        for (x, y) in pontos_referencia:
            cv2.circle(imagem_entrada, (x, y), 1, (255, 255, 0), -1)

        # Exibe a imagem com as marcações
        image_rgb = cv2.cvtColor(imagem_entrada, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.axis('off')  # Opcional: desativa os eixos
        plt.show()

# Chama a função "landmark" passando os rostos detectados
landmark(rects)



