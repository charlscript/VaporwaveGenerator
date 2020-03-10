###
## Este é o código base do programa
## ele recebe a imagem e aplica todos os parametros da A.I 
## após consumir a imagem, e aplicar os padrões neurais treinados e definidos em .XML
## ele retorna a imagem vaporwarizada caso seja possível
###


import random as rd

import cv2
import numpy as np

from .elements import add_single_element, add_elements
from . import mods


def vaporize(image_path):
    face_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('cascade/haarcascade_eye.xml')

    # load main image from local file
    img = cv2.imread(image_path)
    # height, width, depth = img.shape

    # transforma a imagem em preto e branco para detectar face e olhos
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # encontrar todas as faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Iterar por cada uma das faces encontradas
    for face in faces:
        y = face[1]
        x = face[0]
        w = face[2]
        h = face[3]
        roi_gray = gray[y:y + h, x:x + w]
        # Detecção dos olhos para aplicação dos efeitos de cascade
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.2, 6)
        for eye in eyes:
            eye[0] += face[0]
            eye[1] += face[1]

        # aqui a A.I escolhe qual modificação irá aplicar em cada rosto detectado
        eyes_present = len(eyes) >= 2
        mod_function, operates_on = mods.determine_face_mod(eyes_present)
        if operates_on == mods.EYES:
            modded_img = mod_function(img, eyes)
            if modded_img is not None:
                img = modded_img
        elif operates_on == mods.FACE:
            modded_img = mod_function(img, face)
            if modded_img is not None:
                img = modded_img

    # adiciona elementos aleatórios na imagem   
    add_elements(img)
    # se não houver rostos, apenas adicione mais elementos
    if len(faces) < 1:
        add_elements(img)

    # A.I escolhe qual contraste será utilizado
    choice = rd.randint(0, 1)
    if choice == 1:
        # edição de canais alfa e beta para o contraste
        img = cv2.convertScaleAbs(img, alpha=1.2, beta=35)

    # A.I escolhe a quantidade de noise sera utilizada na imagem
    choice = rd.randint(0, 1)
    if choice == 1:
        row, col, ch = img.shape
        mean = 0
        # var = variável de controle de ruído TODO: receber via parâmetro
        var = 15
        sigma = var ** 1
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = (img + gauss)
        cv2.normalize(noisy, noisy, 0, 1, cv2.NORM_MINMAX)
        img = noisy

    # O código abaixo é útil para determinar se olhos ou rostos
    # estão sendo detectados corretamente. Descomentar irá desenhar caixas
    # ao redor das features detectadas.
    # for (x,y,w,h) in faces:
    #     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    #     roi_gray = gray[y:y+h, x:x+w]
    #     roi_color = img[y:y+h, x:x+w]
    #     #edit the second and third parameter if feature detection is poor
    #     eyes = eye_cascade.detectMultiScale(roi_gray, 1.2, 6)
    #     for (ex,ey,ew,eh) in eyes:
    #         cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    return img
