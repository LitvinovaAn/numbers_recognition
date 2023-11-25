import cv2
import numpy as np


def augmentation(img):
    rand = np.random.randint(0, 5)
    new_img = img.copy()
    if rand == 1:
        # Отразит слева-направо
        new_img = cv2.flip(new_img, 1)
    elif rand == 2:
        # Отразит сверху-вниз
        new_img = cv2.flip(new_img, 0)
    elif rand == 3:
        # Повернет на 90 по часовой стрелке
        new_img = cv2.rotate(new_img, cv2.ROTATE_90_CLOCKWISE)
    elif rand == 4:
        # Повернет на 90 градусов против часовой стрелки
        new_img = cv2.rotate(new_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rand == 5:
        # Повернуть на 180
        new_img = cv2.flip(new_img, cv2.ROTATE_180)

    return new_img
