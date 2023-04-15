import cv2
import os


def cut_a_image():
    path = r"..\input_image\1944-1944"
    out_path = r"..\input_image\1944-1944"
    files = os.listdir(path)

    file = "23label.jpg"
    print(file)
    img = cv2.imread(os.path.join(path, file))
    cropped = img[400:, :-400]
    cv2.imwrite(os.path.join(out_path, "23-1label.jpg"), cropped)


if __name__ == "__main__":
    cut_a_image()
