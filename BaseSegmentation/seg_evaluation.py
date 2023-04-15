import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.metrics import f1_score


def evalutation_seg(input, target, explain_str):
    print(explain_str+ "-------begin-----------")
    print(explain_str + "-compute_acc:", round(compute_acc(input, target), 2))
    print(explain_str + "-mean_iou:", round(mean_iou(input, target, classes=2), 2))
    # print(explain_str + "-compute_f1:", round(compute_f1(input, target), 2))
    print(explain_str + "-compute_kappa:", round(compute_kappa(input, target), 2))
    # print(explain_str + "-compute_recall:", np.round(compute_recall(input, target), 2))
    print(explain_str+ "-------end-----------")


def mean_iou(input, target, classes=2):
    """  compute the value of mean iou
    :param input:  2d array, int, prediction
    :param target: 2d array, int, ground truth
    :param classes: int, the number of class
    :return:
        miou: float, the value of miou
    """
    miou = 0
    for i in range(classes):
        intersection = np.logical_and(target == i, input == i)
        # print(intersection.any())
        union = np.logical_or(target == i, input == i)
        temp = np.sum(intersection) / np.sum(union)
        miou += temp
    return miou / classes


def iou(input, target, classes=1):
    """  compute the value of iou
    :param input:  2d array, int, prediction
    :param target: 2d array, int, ground truth
    :param classes: int, the number of class
    :return:
        iou: float, the value of iou
    """
    intersection = np.logical_and(target == classes, input == classes)
    # print(intersection.any())
    union = np.logical_or(target == classes, input == classes)
    iou = np.sum(intersection) / np.sum(union)
    return iou


def compute_f1(prediction, target):
    """
    :param prediction: 2d array, int,
            estimated targets as returned by a classifier
    :param target: 2d array, int,
            ground truth
    :return:
        f1: float
    """
    prediction.tolist(), target.tolist()
    img, target = np.array(prediction).flatten(), np.array(target).flatten()
    f1 = f1_score(y_true=target, y_pred=img)
    return f1


def compute_kappa(prediction, target):
    """
    :param prediction: 2d array, int,
            estimated targets as returned by a classifier
    :param target: 2d array, int,
            ground truth
    :return:
        kappa: float
    """
    prediction.tolist(), target.tolist()
    img, target = np.array(prediction).flatten(), np.array(target).flatten()
    kappa = cohen_kappa_score(target, img)
    return kappa


def compute_acc(pred, gt):
    matrix = confusion_matrix(y_true=np.array(gt).flatten(), y_pred=np.array(pred).flatten())
    acc = np.diag(matrix).sum() / matrix.sum()
    return acc


def compute_recall(pred, gt):
    #  返回所有类别的召回率recall
    matrix = confusion_matrix(y_true=np.array(gt).flatten(), y_pred=np.array(pred).flatten())
    recall = np.diag(matrix) / matrix.sum(axis=0)
    return recall
