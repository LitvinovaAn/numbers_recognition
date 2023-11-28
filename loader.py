from glob import glob
import cv2


def image_label(path):
    images = glob(f'{path}/*.jpg')
    labels = []
    labels_files = [open(file, "r").read() for file in glob(f'{path}/*.txt')]
    for file in labels_files:
        if file == "":
            labels.append('100')
        else:
            labels.append(''.join([x[0] for x in file.split("\n")]))

    return images, labels


def load(path_to_data):
    train_image, train_label = image_label(f'{path_to_data}/train')
    valid_image, valid_label = image_label(f'{path_to_data}/test')

    return train_image, train_label, valid_image, valid_label


if __name__ == '__main__':
    path = 'numbers101'
    train_image, train_label, valid_image, valid_label = load(path)

    for i in range(len(train_image)):
        img = cv2.imread(train_image[i])
        num = train_image[i].split('-')[2].split('.')[0]

        for j in train_label[i].split("\n"):
            img_test = img.copy()
            x, y, w, h = j.split(' ')[1:]
            hx, hy = int(img.shape[0] / 2), int(img.shape[1] / 2)
            x, y = round(float(x)*hx), round(float(y)*hy)
            w, h = round(float(w)*hx/2), round(float(h)*hy/2)

            cv2.rectangle(img_test, (x-w, y-h), (x+w, y+h), (0, 0, 255), 2, 1)
            cv2.imshow('', img_test)
            cv2.waitKey(0)

        img_test = img.copy()
        cv2.putText(img_test, num, (20, 20), 1, 2, (255, 0, 0))
        cv2.imshow('', img_test)
        cv2.waitKey(0)


