import dlib
import cv2
import imutils
import matplotlib.pyplot as plt
from imutils import face_utils

detector_faces = dlib.get_frontal_face_detector()
predictor_pontos = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def landmark_from_webcam():
    # Inicializar a captura de vídeo da webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capturar um frame da webcam
        ret, frame = cap.read()
        
        # Converter o frame para escala de cinza
        frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostos no frame
        rects = detector_faces(frame_cinza)

        # Chamar a função landmark() para processar os rostos
        landmark(rects, frame)

        # Exibir o frame com as marcações
        cv2.imshow("Frame", frame)

        # Verificar se a tecla 'q' foi pressionada para sair do loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar os recursos utilizados
    cap.release()
    cv2.destroyAllWindows()

def landmark(rects, frame):
    for (i, rect) in enumerate(rects):
        pontos_referencia = predictor_pontos(frame, rect)
        pontos_referencia = face_utils.shape_to_np(pontos_referencia)

        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        for (x, y) in pontos_referencia:
            cv2.circle(frame, (x, y), 1, (255, 255, 0), -1)

    # Exibir o frame com as marcações
    cv2.imshow("Frame", frame)

# Chamar a função landmark_from_webcam() para iniciar o processamento da webcam
landmark_from_webcam()
