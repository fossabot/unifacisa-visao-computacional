# ****************************************************************
# AULA: Visão Computacional
# Prof: Adriano A. Santos, DSc.
# ****************************************************************

import cv2
import time

video = './dataset/file_example_MP4_1920_18MG.mp4'

# Instancia objeto de manipulacao de videos
# Obs: se for a camera, defina a posicao. Ex: 0
camera = cv2.VideoCapture(video)

# Enquanto existir frames
while camera.isOpened:
    # Get the current frame and the status
    status, frame = camera.read()

    # Stop condition
    key = cv2.waitKey(1) & 0xFF

    # Controle de execucao
    if (status == False or key == ord("q")):
        break
    
    # Visualiza o frame
    cv2.imshow("Frame", frame)
    time.sleep(0.02) # Controle da velocidade de execucao do video.

camera.release()
cv2.destroyAllWindows()