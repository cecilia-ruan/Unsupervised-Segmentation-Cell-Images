import cv2
import numpy as np
from skimage import segmentation

from BaseSegmentation.base_utils import imshow_image, read_image, process_gauss_laplace

mark_boundaries = segmentation.mark_boundaries

if __name__ == '__main__':

    image_ori_gray = read_image(r"../../cell_images/cut_cell_200_200/cir_D42_YL-2-03.jpg")
    # cir_D42_YL-2-01.jpg 好5
    # cir_D42_YL-2-02.jpg 良3
    # cir_D42_YL-2-03.jpg 好4

    '''resize成小图'''
    input_size = [300, 300]
    img = cv2.resize(image_ori_gray, input_size)
    img_for_felz = img
    paras = [32.06175906594817, 1.7915903999999998, 242.5985430122915]
    paras = [72.13895789838335, 0.5635856085811202, 84.8644673048463]
    gauss_target = cv2.resize(process_gauss_laplace(image_ori_gray), input_size)

    # '''直接截取大图'''
    # img = image_ori_gray#[0:500, 0:500]
    # paras = [92.16, 0.5, 3407.7866599533095]  # 大图
    # img_for_felz = cv2.GaussianBlur(img, (21, 21), 0)  # 大图
    # gauss_target = process_gauss_laplace(img)


    seg_first = segmentation.felzenszwalb(img_for_felz, scale=int(paras[0]),
                                                sigma=paras[1],
                                                min_size=int(paras[2]))
    print("seg_first", seg_first.shape)
    # seg_map = segmentation.slic(test_image, n_segments=10000, compactness=100)
    boundaries = mark_boundaries(img, seg_first)[:, :, 2]
    # imshow_image([img, seg_first, boundaries])

    seg_map_flatten = seg_first.flatten()
    gauss_target_flatten = gauss_target.flatten()
    assert seg_map_flatten.size == gauss_target_flatten.size

    seg_lab = [np.where(seg_map_flatten == u_label)[0]
               for u_label in np.unique(seg_map_flatten)]

    # 把pre_seg中分好类的每一类的位置，看看其在gauss分割图上是0多还是1多。从而决定pre_seg这类位置是0还是1
    for inds in seg_lab:
        u_labels, hist = np.unique(gauss_target_flatten[inds], return_counts=True)
        seg_map_flatten[inds] = u_labels[np.argmax(hist)]

    final = seg_map_flatten.reshape(img.shape)
    imshow_image([img, seg_first, gauss_target, final])





