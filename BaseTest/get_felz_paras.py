import cv2
from skimage import segmentation

from BaseSegmentation.base_utils import normalization, imshow_image, save_image

mark_boundaries = segmentation.mark_boundaries


def press_wasdqe_to_adjust_parameter_of_felz(img):
    paras = [64, 0.5, 128]  # pieces
    paras = [1, 0.8, 20]  # default
    paras = [72.13895789838335, 0.5635856085811202, 84.8644673048463]
    act_dict = {0: (0, 1.0)}
    act_dict.update(zip([119, 97, 113], [(i, 1.2) for i in range(len(paras))], ))  # _KeyBoard: W A Q
    act_dict.update(zip([115, 100, 101], [(i, 0.8) for i in range(len(paras))], ))  # KeyBoard: S D E
    key = 0
    while True:
        if key != -1:
            i, multi = act_dict[key]
            paras[i] *= multi
            print(key, paras)

            seg_map = segmentation.felzenszwalb(img,
                                                scale=int(paras[0]),
                                                sigma=paras[1],
                                                min_size=int(paras[2]))

            print(seg_map.shape)
            show = mark_boundaries(img, seg_map)
            print(show.shape)
            cv2.imshow('', show)

            # imshow_image([seg_map, ])
            wait_time = 1
        else:
            wait_time = 100

        key = cv2.waitKey(wait_time)
        # cv2.imwrite('tiger_felz.jpg', show*255)


def press_wasdqe_to_adjust_parameter_of_slic(img):
    # paras = [100, 1000, 10]  # pieces
    # paras = [100, 10, 10]  # default
    # paras = [16, 64, 6]  # appropriate

    paras = [100.35999999999999, 800.7055196745926, 100]

    act_dict = {0: (0, 1.0)}
    act_dict.update(zip([119, 97, 113], [(i, 1.2) for i in range(len(paras))], ))  # _KeyBoard: W A Q
    act_dict.update(zip([115, 100, 101], [(i, 0.8) for i in range(len(paras))], ))  # KeyBoard: S D E

    key = 0
    while True:
        if key != -1:
            i, multi = act_dict[key]
            paras[i] *= multi
            print(key, paras)

            seg_map = segmentation.slic(img,
                                        compactness=int(paras[0]),
                                        n_segments=int(paras[1]),
                                        max_iter=int(paras[2]), )
            show = mark_boundaries(img, seg_map)
            cv2.imshow('', show)
            imshow_image([seg_map, ])
            wait_time = 1
        else:
            wait_time = 100

        key = cv2.waitKey(wait_time)
    #     break
    # cv2.imwrite('tiger_slic.jpg', show * 255)


def read_image(img_file):
    img = cv2.imread(img_file)
    img = img.transpose((2, 0, 1))[0]
    return img


if __name__ == '__main__':
    image = read_image('../input_image/1944-1944/52.jpg')
    # test_image = cv2.GaussianBlur(test_image, (21, 21), 0)
    image = cv2.resize(image, [448, 448])
    # image = image[0:500, 0:500]
    press_wasdqe_to_adjust_parameter_of_felz(image)
    # press_wasdqe_to_adjust_parameter_of_slic(image)
