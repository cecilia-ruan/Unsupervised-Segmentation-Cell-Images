import copy

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import segmentation, morphology, measure


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def read_image(img_file):
    # img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

    img = cv2.imread(img_file)
    img = img.transpose((2, 0, 1))[0]
    return img


def imshow_image(img, *args, title=""):
    plt.figure()
    img = np.array(img)
    if img.shape[0] > 20:
        if args:
            plt.imshow(img, vmin=args[0], vmax=args[1], cmap="gray")
        else:
            plt.imshow(img, vmin=np.min(img), vmax=np.max(img), cmap="gray")
    else:
        for i in range(len(img)):
            plt.subplot(1, len(img), i + 1)
            plt.imshow(img[i], vmin=np.min(img[i]), vmax=np.max(img[i]), cmap="gray")
    plt.title(title)
    # plt.colorbar()
    plt.show()


def process_gauss_laplace(img):
    gauss_blur = cv2.GaussianBlur(img, (35, 35), 0)
    laplace_edge = cv2.Laplacian(gauss_blur, cv2.CV_64F)
    print("laplace_edge", np.min(laplace_edge), np.max(laplace_edge))
    laplace_edge = np.uint8(np.absolute(laplace_edge))
    # imshow_image([img, laplace_edge])
    laplace_denoise = cv2.medianBlur(laplace_edge, 11)  # range 0-2
    laplace_denoise[np.where(laplace_denoise > 0)] = 1
    # imshow_image([img, laplace_denoise])
    return laplace_denoise


def save_image(image, path, type="cv", dpi=96):
    if type =="cv":
        cv2.imwrite(path, image)
    if type == "plt":
        plt.figure()
        plt.axis('off')
        plt.imshow(image, vmin=np.min(image), vmax=np.max(image), cmap="gray")
        plt.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0)


def felzenszwalb_seg(img_for_felz, paras):
    seg_out = segmentation.felzenszwalb(img_for_felz, scale=int(paras[0]),
                                        sigma=paras[1],
                                        min_size=int(paras[2]))
    return seg_out


def get_combine_seg(args, image_ori):
    image_ori_gray = image_ori.transpose((2, 0, 1))[0]
    if args.is_scaling:
        image = cv2.resize(image_ori, args.input_size)
        image_gray = cv2.resize(image_ori_gray, args.input_size)
    else:
        image = image_ori[0:args.input_size[0], 0:args.input_size[1], :]
        image_gray = image_ori_gray[0:args.input_size[0], 0:args.input_size[1]]
    '''get gauss laplace：都from原图，再resise到指定大小'''
    gauss_seg_map = process_gauss_laplace(image_ori_gray)
    if args.is_scaling:
        gauss_seg_map = cv2.resize(gauss_seg_map, args.input_size)
    else:
        gauss_seg_map = gauss_seg_map[0:args.input_size[0], 0:args.input_size[1]]
    gauss_seg_flatten = gauss_seg_map.flatten()

    '''segmentation felz：'''
    if args.is_scaling:
        img_for_felz = image
        felz_paras = args.felz_paras_s
    else:
        img_for_felz = cv2.GaussianBlur(image, (21, 21), 0)
        felz_paras = args.felz_paras_l
    felz_seg_map = felzenszwalb_seg(img_for_felz, felz_paras)
    felz_seg_flatten = felz_seg_map.flatten()

    '''combine gauss & pre_seg'''
    combine_seg_flatten = copy.deepcopy(felz_seg_flatten)
    felz_seg_index = [np.where(combine_seg_flatten == u_label)[0]
                      for u_label in np.unique(combine_seg_flatten)]
    for idx in felz_seg_index:
        u_labels, hist = np.unique(gauss_seg_flatten[idx], return_counts=True)
        combine_seg_flatten[idx] = u_labels[np.argmax(hist)]
    combine_seg_map = combine_seg_flatten.reshape(args.input_size)

    return combine_seg_map


def remove_small_points(image, threshold_point):
    out_image = copy.deepcopy(image)
    img_label, num = measure.label(image, connectivity=1, return_num=True)  # 输出二值图像中所有的连通域
    props = measure.regionprops(img_label)  # 输出连通域的属性，包括面积等
    for i in range(1, len(props)):
        if props[i].area < threshold_point:
            out_image[np.where(img_label == props[i].label)] = 0
    return out_image


