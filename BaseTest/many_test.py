import copy

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure

from Segmentation.base_utils import imshow_image, read_image, save_image, normalization
from Segmentation.seg_evaluation import evalutation_seg


def process_difference(img):
    line_noise = cv2.GaussianBlur(img, (35, 35), 0) - cv2.GaussianBlur(img, (11, 11), 0)
    line_denoise = cv2.medianBlur(line_noise, 19)
    line_denoise[np.where(line_denoise < 125)] = 0
    line_denoise[np.where(line_denoise >= 125)] = 1
    imshow_image(line_denoise)


def process_gauss_laplace(img):
    gauss_blur = cv2.GaussianBlur(img, (35, 35), 0)
    laplace_edge = cv2.Laplacian(gauss_blur, cv2.CV_64F)
    laplace_edge_abs = np.uint8(np.absolute(laplace_edge))

    laplace_denoise = cv2.medianBlur(laplace_edge_abs, 11)  # range 0-2
    laplace_denoise_bin = copy.deepcopy(laplace_denoise)
    laplace_denoise_bin[np.where(laplace_denoise > 0)] = 1

    imshow_image([img, gauss_blur, laplace_edge, laplace_edge_abs, laplace_denoise, laplace_denoise_bin])

    gauss_blur = normalization(gauss_blur)
    laplace_edge = normalization(laplace_edge)
    laplace_edge_abs = normalization(laplace_edge_abs)
    laplace_denoise = normalization(laplace_denoise)
    laplace_denoise_bin = normalization(laplace_denoise_bin)

    print(np.max(gauss_blur), np.max(laplace_edge), np.min(laplace_edge_abs), np.max(laplace_denoise))

    save_image(gauss_blur * 255, r"../output_image/image15.jpgsize448/2-1-gauss_blur.jpg")
    save_image(laplace_edge * 255, r"../output_image/image15.jpgsize448/2-2-laplace_edge.jpg")
    save_image(laplace_edge_abs * 255, r"../output_image/image15.jpgsize448/2-3-laplace_edge_abs.jpg")
    save_image(laplace_denoise * 255, r"../output_image/image15.jpgsize448/2-4-laplace_denoise.jpg")
    save_image(laplace_denoise_bin * 255, r"../output_image/image15.jpgsize448/2-5-laplace_denoise_bin.jpg")

    return laplace_denoise_bin


def try_mln_filter(img):
    # mean_blur = cv2.blur(img, (15, 15))
    # gauss_blur = cv2.GaussianBlur(img, (21, 21), 0)
    # nlm_blur = cv2.fastNlMeansDenoising(img, h=5, templateWindowSize=7, searchWindowSize=31)
    # res = np.hstack((img, gauss_blur, nlm_blur))
    # imshow_image(res)

    equ = cv2.equalizeHist(img)  # 直方图均衡化
    # mean_blur = cv2.blur(equ, (11, 11))
    # gauss_blur = cv2.GaussianBlur(equ, (35, 35), 0)
    # res = np.hstack((equ, mean_blur, gauss_blur))
    imshow_image(equ)

    # imshow_image([img, mean_blur, gauss_blur])
    # cv2.imwrite("mean_blur.jpg", mean_blur * 255)
    # cv2.imwrite("gauss_blur.jpg", gauss_blur * 255)


def remove_small_points(image, threshold_point):
    out_image = copy.deepcopy(image)
    img_label, num = measure.label(image, connectivity=1, return_num=True)  # 输出二值图像中所有的连通域
    props = measure.regionprops(img_label)  # 输出连通域的属性，包括面积等
    for i in range(1, len(props)):
        if props[i].area < threshold_point:
            out_image[np.where(img_label == props[i].label)] = 0
    return out_image


def deal_background(img_file):
    image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    image[np.where(image > 10)] = 255
    image[np.where(image <= 10)] = 0
    # image = cv2.resize(image, [300, 300])
    imshow_image(image)
    # save_image(image, 'image-2-03size300/0.background.png',  dpi=96)
    cv2.imwrite("../input_image/1944-1944/52 fake back use.jpg", image)


def read_evalutation_seg():
    outputpath = "../output_image/image52.jpgsize448/"
    inputpath = "../input_image/1944-1944/"

    combine_seg_map = cv2.imread(outputpath + "4.combine_seg_map.png", cv2.IMREAD_GRAYSCALE)
    cnn_seg_map = cv2.imread(outputpath + "5.cnn_seg_map.png", cv2.IMREAD_GRAYSCALE)
    target = cv2.resize(read_image(inputpath + "52 fake back use.jpg") // 255, [448, 448])
    evalutation_seg(combine_seg_map, target, "combine_seg_map")
    evalutation_seg(cnn_seg_map, target, "cnn_seg_map")


def deal_background_labelme():
    img_file = "52label.png"
    image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    image[np.where(image > 10)] = 255
    image[np.where(image <= 10)] = 0
    # image = cv2.resize(image, [300, 300])
    imshow_image(image)
    save_image(image, '../input_image/1944-1944/52label.jpg')


def plt_hist():
    img = r"../input_image/1944-1944/background2.png"
    img1 = cv2.resize(read_image(img), [80, 80])
    img = r"../input_image/1944-1944/incell.png"
    img2 = cv2.resize(read_image(img), [80, 80])

    plt.hist(img2.ravel(), bins=100, color="darkorange", alpha=0.5)  # moccasin  navajowhite
    plt.axis('off')
    plt.show()

    bins = np.linspace(100, 160, 100)
    plt.hist(img1.ravel(), bins, alpha=0.5, label='background')
    plt.hist(img2.ravel(), bins, alpha=0.5, label='cell')
    plt.legend(loc='upper right')
    plt.show()


def oppose_binary_image():
    img_file = "6.cnn_seg_map_oppose.jpg"
    image = cv2.imread(img_file)
    image = 255 - image
    # image[np.where(image > 10)] = 255
    # image[np.where(image <= 10)] = 0
    # image = cv2.resize(image, [300, 300])
    imshow_image(image)
    save_image(image, '66.cnn_seg_map_oppose.jpg')


def get_3_channel():
    img_file = "15.jpg"
    image = cv2.imread(img_file)
    # image[np.where(image > 10)] = 255
    # image[np.where(image <= 10)] = 0
    # image = cv2.resize(image, [300, 300])
    imshow_image(image)
    image1 = image.transpose((2, 0, 1))[0]
    image2 = image.transpose((2, 0, 1))[1]
    image3 = image.transpose((2, 0, 1))[2]
    save_image(image1, '15.1.jpg')
    save_image(image2, '15.2.jpg')
    save_image(image3, '15.3.jpg')


def deal_demo_origin_slic():
    img_file = "seg_slic.jpg"
    image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    image[np.where(image > 120)] = 255
    image[np.where(image <= 120)] = 0
    imshow_image(image)
    save_image(image, 'seg_slic_bin.jpg')


def deal_demo_modify_felz():
    img_file = "seg_felz.jpg"
    image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    image[np.where(image > 10)] = 255
    image[np.where(image <= 10)] = 0
    image = 255 - image
    imshow_image(image)
    save_image(image, 'seg_felz_bin.jpg')


def demo_evalutation():
    slic = cv2.imread("seg_slic_bin.jpg", cv2.IMREAD_GRAYSCALE)
    felz = cv2.imread("seg_felz_bin.jpg", cv2.IMREAD_GRAYSCALE)
    target = cv2.imread("0.image_target.jpg", cv2.IMREAD_GRAYSCALE)
    print(slic.shape)
    evalutation_seg(slic, target, "slic")
    evalutation_seg(felz, target, "felz")


if __name__ == '__main__':
    # img = r"../input_image/1944-1944/background.png"
    # img = cv2.resize(read_image(img),[80,80])
    # process_gauss_laplace(img)

    demo_evalutation()
    pass
