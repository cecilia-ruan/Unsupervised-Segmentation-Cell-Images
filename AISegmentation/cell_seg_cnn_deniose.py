import copy
import os
import time
import torch
import numpy as np
import cv2

from AISegmentation.cnn_utils import SegNet
from BaseSegmentation.base_utils import read_image, \
    process_gauss_laplace, felzenszwalb_seg, imshow_image, save_image, remove_small_points, normalization
from BaseSegmentation.seg_evaluation import mean_iou, evalutation_seg


class Args(object):
    path = r"../input_image/1944-1944"
    image_name = "15.jpg"
    target_name = "15label.jpg"
    input_size = [448, 448]
    out_directory = "../output_image/image{}size{}/".format(image_name, input_size[0])

    input_image_path = os.path.join(path, image_name)
    target = cv2.resize(read_image(os.path.join(path, target_name)) / 255, input_size)
    target[np.where(target >= 0.5)] = 1
    target[np.where(target < 0.5)] = 0
    is_scaling = True  # if True:将原图resize，if False:就从原图裁剪到input size
    felz_paras_s = [72.13895789838335, 0.5635856085811202, 84.8644673048463]  # base
    # felz_paras_s = [60.13895789838335, 0.5635856085811202, 60.8644673048463]
    felz_paras_l = [152.8823808, 0.576, 548.8624596091704]
    train_epoch = 20
    mod_dim1 = 64
    mod_dim2 = 32
    out_dim = 2  # 分割类别


def run():
    args = Args()
    target = args.target

    '''read test_image'''
    image_ori = cv2.imread(args.input_image_path)
    image_ori_gray = read_image(args.input_image_path)
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

    '''show and save'''
    # imshow_image([image_gray, gauss_seg_map, felz_seg_map, combine_seg_map])

    # save_image(target*255, '{}/0.image_target.jpg'.format(args.out_directory))
    # save_image(image_gray, '{}/1.image_gray.jpg'.format(args.out_directory))
    # save_image(gauss_seg_map*255, '{}/2.gauss_seg_map.jpg'.format(args.out_directory))
    # print(np.max(felz_seg_map))
    # save_image(normalization(felz_seg_map)*255, '{}/3.felz_seg_map.jpg'.format(args.out_directory))
    # save_image(combine_seg_map*255, '{}/4.combine_seg_map.jpg'.format(args.out_directory))

    # return

    '''train init'''
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    image_tensor = image.transpose((2, 0, 1)).astype(np.float32) / 255.0
    image_tensor = image_tensor[np.newaxis, :, :, :]
    image_tensor = torch.from_numpy(image_tensor).to(device)
    model = SegNet(inp_dim=3, mod_dim1=args.mod_dim1, mod_dim2=args.mod_dim2, out_dim=args.out_dim).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-3, momentum=0.9)

    '''train loop'''
    model.train()
    last_loss = 100.
    for epoch in range(args.train_epoch):
        '''forward'''
        optimizer.zero_grad()
        output = model(image_tensor)[0]  # [2, 300, 300]
        output_flatten = output.permute(1, 2, 0).view(-1, args.out_dim)
        output_flatten_class = torch.argmax(output_flatten, 1).data.cpu().numpy()
        loss = criterion(output_flatten, torch.from_numpy(combine_seg_flatten).to(device))
        loss.backward()
        optimizer.step()

        '''refine'''
        # for idx in felz_seg_index:
        #     u_labels, hist = np.unique(output_flatten_class[idx], return_counts=True)
        #     output_flatten_class[idx] = u_labels[np.argmax(hist)]

        '''show loss'''
        decrease_rate = (last_loss - loss.item()) / last_loss * 100
        last_loss = loss.item()
        print("epoch:", epoch, " loss:", loss.item(), " decrease_rate:", decrease_rate)
        # if decrease_rate < 0.6 and epoch > 15:  #
        #     break

        '''show test_image'''
        cnn_seg_map = output_flatten_class.reshape(args.input_size)
        cv2.imshow("cnn_seg_map", np.uint8(cnn_seg_map * 255 / (len(np.unique(cnn_seg_map)) - 1)))
        if epoch == args.train_epoch - 1:
            cv2.waitKey(0)
        else:
            cv2.waitKey(500)

    # save_image(cnn_seg_map*255, '{}/5.cnn_seg_map.jpg'.format(args.out_directory))
    imshow_image([image_gray, felz_seg_map, combine_seg_map, cnn_seg_map])

    evalutation_seg(gauss_seg_map, target, "gauss_seg_map")
    evalutation_seg(combine_seg_map, target, "combine_seg_map")
    evalutation_seg(cnn_seg_map, target, "cnn_seg_map")


if __name__ == '__main__':
    run()
