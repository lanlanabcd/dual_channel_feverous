# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2020/11/30 21:48
# Description:
from torchnet import meter
import json
class EvalMetric(object):
    def __init__(self):
        self.loss_meter = meter.AverageValueMeter()
        self.acc_meter = meter.AverageValueMeter()

    def meter_reset(self):
        self.loss_meter.reset()
        self.acc_meter.reset()

    def meter_add(self, loss, acc):
        self.loss_meter.add(loss)
        self.acc_meter.add(acc)

    def print_meter(self):
        print("loss:", self.loss_meter.value()[0])
        print("accuracy:", self.acc_meter.value()[0])