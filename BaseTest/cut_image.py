
import cv2
import os

path = r"..\cell_images\cir_D41_YL"
out_path = r"..\cell_images\cut_cell_200_200"

files = os.listdir(path)
file = files[0]

n = 1944//2
for file in files:
    img = cv2.imread(os.path.join(path, file))
    for i in range(2):
        cropped1 = img[0 + i * n:972 + i * n, 0:972]  # 裁剪坐标为[y0:y1, x0:x1]
        cropped2 = img[0 + i * n:972 + i * n, 972:1944]  # 裁剪坐标为[y0:y1, x0:x1]
        cropped3 = img[0 + i * n:972 + i * n, -972:]  # 裁剪坐标为[y0:y1, x0:x1]
        cv2.imwrite(os.path.join(out_path, "cir_D42_YL-" + file[:-4] + "-" + str(i) + "1.jpg"), cropped1)
        cv2.imwrite(os.path.join(out_path, "cir_D42_YL-" + file[:-4] + "-" + str(i) + "2.jpg"), cropped2)
        cv2.imwrite(os.path.join(out_path, "cir_D42_YL-" + file[:-4] + "-" + str(i) + "3.jpg"), cropped3)
