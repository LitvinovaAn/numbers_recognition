from glob import glob
import cv2
import numpy as np

from models import numbers


input_size = (64, 64)
path_to_data = "numbers/test"
images = glob(f"{path_to_data}/*.jpg")
# path_to_data = "test"
# images = glob(f"{path_to_data}/*.*")
model = numbers(input_size + (3,))
weights_path = "numbers.h5"
model.load_weights(weights_path)

for image in images:
    img = cv2.imread(image)
    inp = cv2.resize(img, input_size)
    inp = np.array([inp])
    y = model.predict(inp)[0]
    y = np.where(y == max(y))[0][0]
    cv2.putText(img, f"{y}", (10, 25), 1, 1.4, (255, 0, 0), 3)
    cv2.putText(img, f"{y}", (10, 25), 1, 1.4, (255, 255, 0), 1)
    cv2.imshow("img", img)
    cv2.waitKey(0)

