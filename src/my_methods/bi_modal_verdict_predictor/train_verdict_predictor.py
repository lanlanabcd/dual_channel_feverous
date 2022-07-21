# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2022/1/3 15:57
# Description:

import os
import torch
from tqdm import tqdm

from bi_modal_arg_parser import get_parser

from bi_modal_cls import BiModalCls
import eval_verdict_predictor
from bi_modal_generator import BiModalGenerator

from transformers import AutoTokenizer, RobertaTokenizer
from torch.utils.data import DataLoader
import torch.nn as nn
import bi_modal_config as config

from base_templates import BasePreprocessor
from my_utils import EvalMetric, set_seed, get_optimizer, compute_metrics
from my_utils import print_grad
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from my_utils import generate_data4debug

# os.chdir("{dir}/DCUF_code")

def main():
    args = get_parser().parse_args()
    print(args)
    set_seed(args.seed)

    generate_data4debug(args.data_dir)
    if "small_" in args.data_dir:
        args.force_generate = True
        args.batch_size = 2
        args.gradient_accumulation_steps = 2

    # 1 table 2 text
    args.bert_name_2 = './bert_weights/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli'
    tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
    tokenizer2 = RobertaTokenizer.from_pretrained(args.bert_name_2, model_max_length=512)

    preprocessor = BasePreprocessor(args)

    args.label2id = config.label2idx
    args.id2label = dict(zip([value for _, value in args.label2id.items()], [key for key, _ in args.label2id.items()]))
    args.config = config
    args.tokenizer = [tokenizer, tokenizer2]

    data_generator = BiModalGenerator
    model = BiModalCls(args)

    train_meter = EvalMetric()
    train_data, valid_data, test_data = preprocessor.process(
        args.data_dir, args.cache_dir, data_generator, args.tokenizer, dataset=["train", "dev"])

    collate_fn = data_generator.collate_fn()

    train_dataloader = DataLoader(train_data, args.batch_size, collate_fn=collate_fn, shuffle=True)
    val_dataloader = DataLoader(valid_data, args.batch_size, collate_fn=collate_fn, shuffle=False)
    if test_data:
        test_dataloader = DataLoader(test_data, args.batch_size, collate_fn=collate_fn, shuffle=False)

    if args.load_model_path:
        model.load(args.load_model_path)

    model.to(args.device)
    tb = SummaryWriter()
    optimizer = get_optimizer(model, lr=args.lr, weight_decay=args.weight_decay, fix_bert=False)

    total_steps = int(args.max_epoch * len(train_dataloader)) // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer
                                                , num_warmup_steps=int(total_steps * args.warm_rate)
                                                , num_training_steps=total_steps)

    criterion = nn.NLLLoss()
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(args.max_epoch):
        model.train()
        loss_sum = 0
        acc_sum = 0
        train_dataloader.dataset.print_example()
        for ii, batch in tqdm(enumerate(train_dataloader)):
            # train model
            try:
                res = model(batch, args, test_mode=False)
            except RuntimeError as e:
                torch.cuda.empty_cache()
                res = model(batch, args, test_mode=False)

            # logits [batch_size, num_classes]
            if len(res) == 2:
                logits, golds = res
                loss = criterion(logits, golds)
            elif len(res) == 3:
                logits, golds, apd_loss = res
                loss = criterion(logits, golds) + apd_loss
            else:
                assert False

            preds = logits.topk(k=1, dim=-1)[1].squeeze(-1).cpu()
            golds = golds.cpu()
            assert len(preds) == len(golds)
            acc = sum([1 if p == g else 0 for p, g in zip(preds, golds)]) / len(preds)

            train_meter.meter_add(loss.item(), acc)

            global_step += 1
            acc_sum += acc * len(preds)
            loss = loss / args.gradient_accumulation_steps
            tb.add_scalar("train_loss", loss.item(), global_step)
            tb.add_scalar("train_acc", acc, global_step)
            loss.backward()

            if global_step % args.gradient_accumulation_steps == 0:

                lrs = scheduler.get_last_lr()
                tb.add_scalars("learning_rates", {"bert_lr": lrs[0], "no_bert_lr": lrs[-1]}, global_step)
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=5.0)
                optimizer.step()
                scheduler.step()

                grad_dict_first = print_grad(model)
                tb.add_scalars("model_grads_first", grad_dict_first, global_step)
                '''
                grad_dict_second = print_grad(model, "second")
                tb.add_scalars("model_grads_second", grad_dict_second, global_step)
                '''
                optimizer.zero_grad()

            if ii % args.print_freq == 0:
                print("====epoch {}, step{}===".format(epoch, ii))
                train_meter.print_meter()
                train_meter.meter_reset()

        print("====train step of epoch {} ==========".format(epoch))
        loss = loss_sum / train_dataloader.dataset.__len__()
        acc = acc_sum / train_dataloader.dataset.__len__()
        print_res(loss, acc, "train", epoch)

        # validate
        print("====validation step of epoch {}======================".format(epoch))
        val_loss, val_acc = val(model, val_dataloader, criterion, tokenizer, preprocessor, tb, epoch, args)
        print_res(val_loss, val_acc, "valid", epoch)

        ckpt_meta = {
            "train_loss": loss,
            "val_loss": val_loss,
            "train_acc": acc,
            "val_acc": val_acc,
            "data_generator": args.data_generator,
        }

        path = model.save(args.ckpt_root_dir, ckpt_meta, val_acc, only_max=(not args.save_all_ckpt))
        if path:
            tokenizer.save_vocabulary(path)
            try:
                model.config.save_pretrained(path)
            except:
                pass
            if args.test:
                # TODO: save test results
                print("==============test step of epoch {}=================".format(epoch))
                test_loss, test_acc = val(model, test_dataloader, criterion, tokenizer, preprocessor, tb, epoch, args)
    tb.flush()
    tb.close()
    args.test_ckpt = model.subdir
    print("start testing, the testing checkpoint is", args.test_ckpt)

    eval_verdict_predictor.main(args)

@torch.no_grad()
def val(model, dataloader, criterion, tokenizer, preprocessor, tb, epoch, args):
    """
    计算模型在验证集上的准确率等信息
    """
    dataloader.dataset.print_example()
    loss_sum = 0
    acc_sum = 0
    preds_epoch = []
    golds_epoch = []

    model.eval()
    for ii, data_entry in tqdm(enumerate(dataloader)):
        res = model(data_entry, args, test_mode=True)
        if len(res) == 2:
            logits, golds = res
            loss = criterion(logits, golds)
        elif len(res) == 3:
            logits, golds, apd_loss = res
            loss = criterion(logits, golds) + apd_loss
        else:
            assert False
        preds = logits.topk(k=1, dim=-1)[1].squeeze(-1)
        acc = sum([1 if p == g else 0 for p, g in zip(preds, golds)]) / len(preds)

        loss_sum += loss.item() * len(preds)
        acc_sum += acc * len(preds)

        preds_epoch.extend(list(preds.cpu().numpy().tolist()))
        golds_epoch.extend(list(golds.cpu().numpy().tolist()))

    scores = compute_metrics(preds_epoch, golds_epoch)

    los = loss_sum / dataloader.dataset.__len__()
    acc = acc_sum / dataloader.dataset.__len__()

    tb.add_scalar("val_loss", los, epoch)
    tb.add_scalar("val_acc", acc, epoch)

    return los, acc


def print_res(los, acc, data_type, epoch):
    print("{}_epoch{}_loss:".format(data_type, epoch), los)
    print("{}_epoch{}_accuracy:".format(data_type, epoch), acc)


if __name__ == "__main__":
    main()
