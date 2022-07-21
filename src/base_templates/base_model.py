# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2020/12/1 12:22
# Description:
import torch.nn as nn
import torch
import time
import json
import os


class BasicModule(nn.Module):
    '''
    封装了nn.Module，主要提供save和load两个方法
    '''

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self)).split(".")[-1][:-2]
        self.previous_log = None
        self.previous_checkpoint = None
        self.last_max = -1
        self.subdir = None

    def load(self, path):
        print("loading checkpoint from: {}".format(path))
        save_dict = torch.load(path + "/pytorch_model.bin", map_location=lambda storage, loc: storage)
        state_dict = save_dict['state_dict']
        ckpt_meta = save_dict['ckpt_meta']
        other_keys = self.load_state_dict(state_dict, strict=True)
        # self.count_parameters()
        return ckpt_meta

    def modify_pretrain_weights(self, state_dict, word_indexes, new_word_indexes):
        assert False, "BasicModule::modify_pretrain_weights ,you shouldn't go here!"

    def save(self, ckpt_root_dir, ckpt_meta, valid_val, only_max=True):
        if self.subdir == None or only_max is False:
            self.subdir = "{}_{}".format(self.model_name, time.strftime('%m%d_%H:%M:%S'))
            path = os.path.join(ckpt_root_dir, self.subdir)
            os.mkdir(path)
        else:
            path = os.path.join(ckpt_root_dir, self.subdir)

        if valid_val > self.last_max or only_max is False:
            if only_max and self.previous_checkpoint != None:
                os.system("rm -r {}".format(self.previous_checkpoint))
            print("save to checkpoint: {}".format(path))
            if not os.path.exists(path):
                os.mkdir(path)
            self.previous_checkpoint = path
            self.last_max = valid_val
            save_dict = {"ckpt_meta": ckpt_meta, "state_dict": self.state_dict()}
            torch.save(save_dict, os.path.join(path, "pytorch_model.bin"))
            self.previous_checkpoint = path
            return path
        return None

    def save_log(self, curve_list, keystr, name=None):
        '''
        保存绘制曲线的数据
        保存模型，默认使用“模型名字+时间”作为文件名，
        如AlexNet_0710_23:57:29.pth
        '''
        pass

    def create_mask(self, src):
        mask = (src != self.args.tokenizer.pad_token_id)
        return mask

    def init_optimizer_n_loss(self):
        assert False, "BasicModule::init_optimizer_n_loss,you shouldn't go here!"

    def train_step(self, **input):
        assert False, "BasicModule::train_step ,you shouldn't go here!"

    def eval_step(self, **input):
        assert False, "BasicModule::eval_step ,you shouldn't go here!"

    def get_optimizers(self):
        assert False, "BasicModule::get_optimizers ,you shouldn't go here!"

    def init_weights(self):
        for idx, (name, param) in enumerate(self.named_parameters()):
            print(str(idx), end=' -> ')
            if ('weight' in name) and param.requires_grad == True:
                print("init", end=' ')
                # nn.init.normal_(param.data, mean=0, std=0.01)
                # nn.init.xavier_uniform_(param.data)
                # nn.init.kaiming_uniform_(param.data)
                # try:
                #     nn.init.kaiming_normal_(param.data)
                # except:
                #     pass
                nn.init.xavier_normal_(param.data)
            else:
                print("not init", end=' ')
                nn.init.constant_(param.data, 0)

            print(name, param.shape)

    def count_parameters(self):
        num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {num:,} trainable parameters')

    def print_modules(self):
        for idx, m in enumerate(self.modules()):
            print(idx, '->', m)

if __name__ == "__main__":
    model = BasicModule()
