from glob import glob
import cv2
import numpy as np

from models import numbers

path_to_data = "numbers101/test"
images = glob(f"{path_to_data}/*.jpg")
model = numbers((80, 80, 3))
weights_path = "numbers.h5"
model.load_weights(weights_path)

for image in images:
    img = cv2.imread(image)
    inp = cv2.resize(img, (80, 80))
    inp = np.array([inp])
    y = model.predict(inp)[0]
    y = np.where(y == max(y))[0][0]
    cv2.putText(img, f"{y}", (10, 25), 1, 1.4, (255, 0, 0), 3)
    cv2.putText(img, f"{y}", (10, 25), 1, 1.4, (255, 255, 0), 1)
    cv2.imshow("img", img)
    cv2.waitKey(0)

