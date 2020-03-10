import math

import cv2
import logging
import numpy as np
import random as rd

NO_MOD = 0
FACE_GLITCH = 1
FACE_DRAG = 2
EYE_CENSOR = 3
EYE_DRAG = 4

EYES = 100
FACE = 101

DEGREES_PER_RADIAN = 57.296

TOP_LEFT = 0
TOP_RIGHT = 1
BOTTOM_RIGHT = 2
BOTTOM_LEFT = 3

logger = logging.getLogger("mods")


def sort_corners(corners):
    corners = sorted(corners, key=lambda x: x[0])
    top_left, bottom_left = sorted(corners[0:2], key=lambda x: x[1])
    top_right, bottom_right = sorted(corners[2:4], key=lambda x: [1])

    return top_left, top_right, bottom_right, bottom_left


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle(first, second, to_degrees=True):
    unit_first = unit_vector(first)
    unit_second = unit_vector(second)

    radians = np.arccos(np.clip(np.dot(unit_first, unit_second), -1.0, 1.0))
    return radians * DEGREES_PER_RADIAN if to_degrees else radians


def pos_and_angle(pts):
    left_upper = pts[TOP_LEFT]
    right_upper = pts[TOP_RIGHT]

    vector = np.array(left_upper - right_upper)
    y_axis = np.array(np.array([0, 0]) - np.array([1, 0]))
    return left_upper, angle(vector, y_axis)


def determine_face_mod(eyes_present):
    function_list = [
        (lambda x, y: x, FACE),
        (face_glitch, FACE),
        (face_drag, FACE),
        (eye_censor, EYES),
        (eye_drag, EYES)
    ]

    function_index = rd.randint(0, 4) if eyes_present else rd.randint(0, 2)

    return function_list[function_index]


def eye_drag(img, eyes):
    # If para ter certeza que apenas dois olhos foram detectados em um rosto
    if len(eyes) > 2:
        eye1 = eyes[0]
        eye2 = eyes[0]
        size = 0
        for itr in range(0, len(eyes)):
            if eyes[itr][2] * eyes[itr][3] > size:
                size = eyes[itr][2] * eyes[itr][3]
                eye1 = eyes[itr]
        size = 0
        for itr in range(0, len(eyes)):
            if eyes[itr][2] * eyes[itr][3] > size and not np.array_equal(eyes[itr], eye1):
                size = eyes[itr][2] * eyes[itr][3]
                eye2 = eyes[itr]
        eyes = [eye1, eye2]

    # a partir daqui apenas rostos com dois olhos serão utilizados
    # isso é importante pois fotos com dois rostos um do lado do outro podem ser erroneamente detectados
    for eye in eyes:
        # acha a largura de cada olho na foto
        iwid = eye[2]
        strp = int(round(iwid / 20.))
        num_glitches = int(eye[2] / strp)
        line = rd.randint(1, eye[3])
        line += eye[1]
        line = int(eye[1] + eye[3] / 2)
        for itr in range(0, num_glitches):
            drop = rd.randint(10, 200)
            if line + drop > img.shape[0]:
                drop = img.shape[0] - line
            img[line:line + drop, eye[0] + itr * strp:eye[0] + itr * strp + strp] = \
                img[line, eye[0] + itr * strp:eye[0] + itr * strp + strp]


def eye_censor(img, eyes):
    if len(eyes) < 2:
        logger.warning("Falha ao gerar o censor, menos de dois olhos detectados.")
        return
    centroid_right = np.array([eyes[0][0] + eyes[0][2] / 2.0, eyes[0][1] + eyes[0][3] / 2.0])
    centroid_left = np.array([eyes[1][0] + eyes[1][2] / 2.0, eyes[1][1] + eyes[1][3] / 2.0])
    vec = centroid_right - centroid_left
    vec = vec / (vec[0] ** 2.0 + vec[1] ** 2.0) ** 0.5
    per_vec = np.array([vec[1], vec[0] * (-1)])
    w_ex = 40
    mag = 75
    right_upper = per_vec * w_ex + centroid_right
    right_lower = centroid_right - per_vec * w_ex
    left_upper = per_vec * w_ex + centroid_left
    left_lower = centroid_left - per_vec * w_ex
    right_upper += vec * mag
    right_lower += vec * mag
    left_upper -= vec * mag
    left_lower -= vec * mag
    corners = sort_corners([right_upper, right_lower, left_lower, left_upper])
    print(corners)
    cv2.fillPoly(img, np.array([corners], dtype=np.int32), (0, 0, 0))

    should_render_text = rd.randint(0, 2)
    if should_render_text:
        with open("elements/censor.txt", "r") as text_file:
            allText = text_file.read()
            possText = allText.split(";")
            dec = rd.randint(0, len(possText) - 1)
            text = possText[dec]
            # calculate text position and angle
            return render_text(text, corners, img)


def render_text(text, corners, img):
    left_upper, right_upper, right_lower, left_lower = corners
    corner, rotation_angle = pos_and_angle(corners)

    text_image = np.ones(img.shape)

    text_img_rows, text_img_cols, _ = text_image.shape

    font = cv2.FONT_HERSHEY_SIMPLEX

    text_size = cv2.getTextSize(text, font, 1, 1)
    (text_width, text_height), _ = text_size

    text_corner_x = left_upper[0] + (right_upper[0] - left_upper[0]) / 2.0 - text_width / 2.0

    text_corner_y = left_upper[1] + (left_lower[1] - left_upper[1]) / 2.0 - text_height / 2.0

    corner_coords = (int(text_corner_x), int(text_corner_y))

    rotation_matrix = cv2.getRotationMatrix2D(corner_coords, rotation_angle, 1)

    cv2.putText(text_image, text, corner_coords, font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    text_image = cv2.warpAffine(text_image, rotation_matrix, (text_img_cols, text_img_rows))

    img = text_image + img
    cv2.imshow("pic", img)
    return img


def face_drag(img, face):
    h, w, d = img.shape
    ornt = rd.randint(0, 2)
    if ornt == 0:
        line = rd.randint(face[1] + 25, face[1] + face[3] - 25)
        direction = rd.randint(0, 2)
        if direction == 0:
            img[0:line, face[0]:face[0] + face[2]] = img[line, face[0]:face[0] + face[2]]
        else:
            img[line:h, face[0]:face[0] + face[2]] = img[line, face[0]:face[0] + face[2]]
    else:
        line = rd.randint(face[0] + 25, face[0] + face[2] - 25)
        direction = rd.randint(0, 2)
        if direction == 0:
            img[face[1]:face[1] + face[3], 0:line] = img[face[1]:face[1] + face[3], line:line + 1]
        else:
            img[face[1]:face[1] + face[3], line:w] = img[face[1]:face[1] + face[3], line:line + 1]


def face_glitch(img, face):
    height, width, d = img.shape
    div = rd.randint(10, 100)
    strp = int(round(face[3] / (div * 1.0)))
    num_glitches = face[3] / strp
    if type(num_glitches) == np.float64:
        num_glitches = math.floor(num_glitches)
    for itr in range(0, num_glitches):
        st_y = face[1] + (itr * strp)
        end_y = face[1] + (itr * strp + strp)
        rng = rd.randint(15, 100)
        dec = rd.randint(1, 2)
        diff = 0

        if dec % 2 == 0:
            if face[0] + face[2] + rng >= width:
                diff = face[0] + face[2] + rng - width
                rng = width - (face[0] + face[2])
            img[st_y:end_y, (face[0] + rng):face[0] + face[2] + rng] = \
                img[st_y:end_y, face[0]:face[0] + face[2] - diff]
        else:
            if face[0] - rng < 0:
                diff = abs(face[0] - rng)
                rng = face[0]
            img[st_y:end_y, (face[0] - rng):face[0] + face[2] - rng] = \
                img[st_y:end_y, face[0]:face[0] + face[2] - diff]
