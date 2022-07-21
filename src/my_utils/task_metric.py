# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2021/7/6 13:17
# Description:
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report


def compute_metrics(preds, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    # class_rep = classification_report(labels, preds, target_names= ['NOT ENOUGH INFO', 'SUPPORTS', 'REFUTES'], output_dict=False)
    class_rep = classification_report(labels, preds, target_names=['NOTSEL', 'SELECT'],
                                      output_dict=False)
    print(class_rep)
    print(acc, recall, precision, f1)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'class_rep': class_rep
    }
