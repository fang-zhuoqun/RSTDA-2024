import numpy as np
from operator import truediv
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, classification_report


def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return average_acc, each_acc


def acc_reports(y_gt, y_pred):
    # target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture', 'Grass-trees',
    #                 'Grass-pasture-mowed', 'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
    #                 'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives', 'Stone-Steel-Towers']
    # classification = classification_report(y_gt, y_pred, digits=4, target_names=target_names)
    # classification = classification_report(y_gt, y_pred, digits=4)
    oa = accuracy_score(y_gt, y_pred)
    confusion = confusion_matrix(y_gt, y_pred)
    aa, each_acc = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_gt, y_pred)

    return oa * 100, aa * 100, kappa * 100, each_acc * 100
