# author = liuwei
# date = 2022-06-28

import logging
import os
import json
import pickle
import math
import random
import time
import datetime
from tqdm import tqdm, trange

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader

from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import RobertaConfig, RobertaTokenizer
from utils import cal_acc_f1_score_with_ids, cal_acc_f1_score_per_label, labels_from_file
from task_dataset import AdversarialDataset
from models import AdversarialModelForRelCls

# set logger, print to console and write to file
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
BASIC_FORMAT = "%(asctime)s:%(levelname)s: %(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
chlr = logging.StreamHandler()  # 输出到控制台的handler
chlr.setFormatter(formatter)
logger.addHandler(chlr)

# for output
dt = datetime.datetime.now()
TIME_CHECKPOINT_DIR = "checkpoint_{}-{}-{}_{}:{}".format(dt.year, dt.month, dt.day, dt.hour, dt.minute)
PREFIX_CHECKPOINT_DIR = "checkpoint"

def get_argparse():
    parser = argparse.ArgumentParser()

    # path
    parser.add_argument("--data_dir", default="data/dataset", type=str)
    parser.add_argument("--dataset", default="pdtb2", type=str, help="pdtb2, pdtb3")
    parser.add_argument("--output_dir", default="data/result", type=str)
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str)
    parser.add_argument("--fold_id", default=-1, type=int, help="-1, 1 to 12")

    # hyperparameters
    parser.add_argument("--relation_type", default="implicit", type=str)
    parser.add_argument("--label_file", default="labels_level_1.txt", type=str, help="the label file path")

    # for training
    parser.add_argument("--do_train", default=False, action="store_true")
    parser.add_argument("--do_dev", default=False, action="store_true")
    parser.add_argument("--do_test", default=False, action="store_true")
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--num_init_epochs", default=5, type=int, help="init stage traning epoch")
    parser.add_argument("--num_train_epochs", default=10, type=int, help="training epoch")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="learning rate")
    parser.add_argument("--max_grad_norm", default=2.0, type=float)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--warmup_ratio", default=0.06, type=float)
    parser.add_argument("--seed", default=106524, type=int, help="random seed")

    return parser

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_dataloader(dataset, args, mode="train"):
    print("{} dataset length: ".format(mode), len(dataset))
    if mode.lower() == "train":
        sampler = RandomSampler(dataset)
        batch_size = args.train_batch_size
    else:
        sampler = SequentialSampler(dataset)
        batch_size = args.eval_batch_size

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=sampler
    )

    return data_loader

def get_optimizer(model, args, num_training_steps):
    specific_params = []
    no_deday = ["bias", "LayerNorm.weigh"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_deday)],
            "weight_decay": args.weight_decay
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_deday)],
            "weight_decay": 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(num_training_steps * args.warmup_ratio),
        num_training_steps=num_training_steps
    )
    return optimizer, scheduler

def train_only_rel(model, args, train_dataset, dev_dataset, test_dataset, label_list, tokenizer, use_conn=False):
    ## 1. prepare data
    train_dataloader = get_dataloader(train_dataset, args, mode="train")
    t_total = int(len(train_dataloader) * args.num_train_epochs)
    num_train_epochs = args.num_init_epochs
    print_step = int(len(train_dataloader) // 4)

    ## 2.optimizer
    optimizer, scheduler = get_optimizer(model, args, t_total)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader.dataset))
    logger.info("  Num Epochs = %d", num_train_epochs)
    logger.info("  Batch size per device = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss = 0.0
    logging_loss = 0.0
    train_iterator = trange(1, int(num_train_epochs) + 1, desc="Epoch")
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        model.train()
        model.zero_grad()
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            ori_inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[4],
                "flag": "Train"
            }
            ori_outputs = model.origin_rel_forward(**ori_inputs)
            arg_inputs = {
                "input_ids": batch[2],
                "attention_mask": batch[3],
                "labels": batch[4],
                "flag": "Train"
            }
            arg_outputs = model.conn_arg_rel_forward(**arg_inputs)
            loss = ori_outputs[0] + arg_outputs[0]

            optimizer.zero_grad()
            loss.backward()
            logging_loss = loss.item()
            tr_loss += logging_loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1
            if global_step % print_step == 0:
                print(" Init Rel, current loss=%.4f, global average loss=%.4f" % (logging_loss, tr_loss / global_step))

        # evaluation and save
        model.eval()
        # output_dir = os.path.join(args.rel_output_dir, TIME_CHECKPOINT_DIR)
        output_dir = os.path.join(args.rel_output_dir, "model")
        output_dir = os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}_{epoch}")
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

def train_only_disc(model, args, train_dataset):
    ## 1. prepare data
    train_dataloader = get_dataloader(train_dataset, args, mode="train")
    t_total = int(len(train_dataloader) * args.num_train_epochs)
    num_train_epochs = args.num_init_epochs
    print_step = int(len(train_dataloader) // 4)

    ## 2.optimizer
    optimizer, scheduler = get_optimizer(model, args, t_total)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader.dataset))
    logger.info("  Num Epochs = %d", num_train_epochs)
    logger.info("  Batch size per device = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss = 0.0
    logging_loss = 0.0
    train_iterator = trange(1, int(num_train_epochs) + 1, desc="Epoch")
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        model.train()
        model.zero_grad()
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "arg_input_ids": batch[2],
                "arg_attention_mask": batch[3],
                "flag": "Train"
            }
            outputs = model.discriminator_forward(**inputs)
            loss = outputs[0]
            optimizer.zero_grad()
            loss.backward()
            logging_loss = loss.item()
            tr_loss += logging_loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1
            if global_step % print_step == 0:
                print(" Init disc, current loss=%.4f, global average loss=%.4f" % (logging_loss, tr_loss / global_step))

        # evaluation and save
        model.eval()
        # output_dir = os.path.join(args.disc_output_dir, TIME_CHECKPOINT_DIR)
        output_dir = os.path.join(args.disc_output_dir, "model")
        output_dir = os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}_{epoch}")
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

def evaluate_only_rel(model, args, dataset, label_list, tokenizer, epoch, desc="dev", use_conn=False):
    dataloader = get_dataloader(dataset, args, mode=desc)

    all_input_ids = None
    all_label_ids = None
    all_predict_ids = None
    all_possible_label_ids = None
    for batch in tqdm(dataloader, desc=desc):
        if use_conn:
            batch_data = (batch[2], batch[3], batch[4], batch[5])
        else:
            batch_data = (batch[0], batch[1], batch[4], batch[5])
        batch = tuple(t.to(args.device) for t in batch_data)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2],
            "flag": "Eval"
        }

        with torch.no_grad():
            if use_conn:
                outputs = model.conn_arg_rel_forward(**inputs)
            else:
                outputs = model.origin_rel_forward(**inputs)
            preds = outputs[0]

        input_ids = batch[0].detach().cpu().numpy()
        label_ids = batch[2].detach().cpu().numpy()
        possible_label_ids = batch[3].detach().cpu().numpy()
        pred_ids = preds.detach().cpu().numpy()
        if all_label_ids is None:
            all_input_ids = input_ids
            all_label_ids = label_ids
            all_predict_ids = pred_ids
            all_possible_label_ids = possible_label_ids
        else:
            all_input_ids = np.append(all_input_ids, input_ids, axis=0)
            all_label_ids = np.append(all_label_ids, label_ids)
            all_predict_ids = np.append(all_predict_ids, pred_ids)
            all_possible_label_ids = np.append(all_possible_label_ids, possible_label_ids, axis=0)

    acc, f1 = cal_acc_f1_score_with_ids(
        pred_ids=all_predict_ids,
        label_ids=all_label_ids,
        possible_label_ids=all_possible_label_ids
    )
    return acc, f1

def joint_train(model, args, train_dataset, dev_dataset, test_dataset, label_list, tokenizer):
    ## 1. prepare data
    train_dataloader = get_dataloader(train_dataset, args, mode="train")
    t_total = int(len(train_dataloader) * args.num_train_epochs)
    num_train_epochs = args.num_train_epochs
    print_step = int(len(train_dataloader) // 4)

    ## 2.optimizer
    optimizer, scheduler = get_optimizer(model, args, t_total)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader.dataset))
    logger.info("  Num Epochs = %d", num_train_epochs)
    logger.info("  Batch size per device = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    dis_tr_loss = 0.0
    dis_logging_loss = 0.0
    rel_tr_loss = 0.0
    rel_logging_los = 0.0
    best_dev = 0.0
    best_dev_epoch = 0
    best_test = 0.0
    best_test_epoch = 0
    res_list = []
    train_iterator = trange(1, int(num_train_epochs) + 1, desc="Epoch")
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        model.train()
        model.zero_grad()
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)

            ## stage 1, training discriminator
            model.set_roberta_ori(do_fix=True)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "arg_input_ids": batch[2],
                "arg_attention_mask": batch[3],
                "flag": "Train"
            }
            outputs = model.discriminator_forward(**inputs)
            loss = outputs[0]
            optimizer.zero_grad()
            loss.backward()
            dis_logging_loss = loss.item()
            dis_tr_loss += dis_logging_loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            ## stage 2, training relation models
            model.set_roberta_ori(do_fix=False)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[4],
                "flag": "Train"
            }
            outputs = model.joint_forward(**inputs)
            loss = outputs[0]
            optimizer.zero_grad()
            loss.backward()
            rel_logging_loss = loss.item()
            rel_tr_loss += rel_logging_loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            scheduler.step()
            global_step += 1

            if global_step % print_step == 0:
                print(
                    " Discriminator: loss=%.4f, global average loss=%.4f; relation: loss=%.4f, global average loss=%.4f" \
                    %(dis_logging_loss, dis_tr_loss / global_step, rel_logging_loss, rel_tr_loss / global_step)
                )

        # evaluation and save
        model.eval()
        # train_acc, train_f1 = joint_evaluate(
        #     model, args, train_dataset, label_list,
        #     tokenizer, epoch, desc="train"
        # )
        dev_acc, dev_f1 = joint_evaluate(
            model, args, dev_dataset, label_list,
            tokenizer, epoch, desc="dev"
        )
        test_acc, test_f1 = joint_evaluate(
            model, args, test_dataset, label_list,
            tokenizer, epoch, desc="test"
        )
        res_list.append((dev_acc, dev_f1, test_acc, test_f1))
        print(" Joint train epoch=%d"%(epoch))
        # print(" Train acc=%.4f, f1=%.4f" % (train_acc, train_f1))
        print(" Dev acc=%.4f, f1=%.4f" % (dev_acc, dev_f1))
        print(" Test acc=%.4f, f1=%.4f" % (test_acc, test_f1))
        if dev_acc+dev_f1 > best_dev:
            best_dev = dev_acc + dev_f1
            best_dev_epoch = epoch
        if test_acc+test_f1 > best_test:
            best_test = test_acc + test_f1
            best_test_epoch = epoch

        # output_dir = os.path.join(args.joint_output_dir, TIME_CHECKPOINT_DIR)
        output_dir = os.path.join(args.joint_output_dir, "model")
        output_dir = os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}_{epoch}")
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    print(" Best dev: epoch=%d, acc=%.4f, f1=%.4f" % (
        best_dev_epoch, res_list[best_dev_epoch-1][0], res_list[best_dev_epoch-1][1])
    )
    print(" Best test: epoch=%d, acc=%.4f, f1=%.4f\n" % (
        best_test_epoch, res_list[best_test_epoch-1][2], res_list[best_test_epoch-1][3])
    )

def joint_evaluate(model, args, dataset, label_list, tokenizer, epoch, desc="dev", write_file=False):
    dataloader = get_dataloader(dataset, args, mode=desc)

    all_input_ids = None
    all_label_ids = None
    all_predict_ids = None
    all_possible_label_ids = None
    for batch in tqdm(dataloader, desc=desc):
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[4],
            "flag": "Eval"
        }

        with torch.no_grad():
            outputs = model.joint_forward(**inputs)
            preds = outputs[0]

        input_ids = batch[0].detach().cpu().numpy()
        label_ids = batch[4].detach().cpu().numpy()
        possible_label_ids = batch[5].detach().cpu().numpy()
        pred_ids = preds.detach().cpu().numpy()
        if all_label_ids is None:
            all_input_ids = input_ids
            all_label_ids = label_ids
            all_predict_ids = pred_ids
            all_possible_label_ids = possible_label_ids
        else:
            all_input_ids = np.append(all_input_ids, input_ids, axis=0)
            all_label_ids = np.append(all_label_ids, label_ids)
            all_predict_ids = np.append(all_predict_ids, pred_ids)
            all_possible_label_ids = np.append(all_possible_label_ids, possible_label_ids, axis=0)
    """
    _ = cal_acc_f1_score_per_label(
        pred_ids=all_predict_ids,
        label_ids=all_label_ids,
        possible_label_ids=all_possible_label_ids,
        label_list=label_list
    )
    """

    acc, f1 = cal_acc_f1_score_with_ids(
        pred_ids=all_predict_ids,
        label_ids=all_label_ids,
        possible_label_ids=all_possible_label_ids
    )
    if write_file:
        all_labels = [label_list[int(idx)] for idx in all_label_ids]
        all_predictions = [label_list[int(idx)] for idx in all_predict_ids]
        all_input_texts = [
            tokenizer.decode(all_input_ids[i], skip_special_tokens=True) for i in range(len(all_input_ids))
        ]
        pred_dir = os.path.join(args.data_dir, "preds")
        os.makedirs(pred_dir, exist_ok=True)
        file_name = os.path.join(pred_dir, "adv+{}_l{}+{}+{}.txt".format(
            desc, args.label_level, epoch, args.seed))
        error_num = 0
        with open(file_name, "w", encoding="utf-8") as f:
            f.write("%-16s\t%-16s\t%s\n" % ("Label", "Pred", "Text"))
            for label, pred, text in zip(all_labels, all_predictions, all_input_texts):
                if label == pred:
                    f.write("%-16s\t%-16s\t%s\n" % (label, pred, text))
                else:
                    error_num += 1
                    f.write("%-16s\t%-16s\t%s\n" % (label, pred, str(error_num) + " " + text))

    return acc, f1

def main():
    args = get_argparse().parse_args()
    if torch.cuda.is_available():
        args.n_gpu = 1
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        args.n_gpu = 0
    args.device = device
    logger.info("Training/evaluation parameters %s", args)
    set_seed(args.seed)

    ## 1. prepare data
    data_dir = os.path.join(args.data_dir, args.dataset)
    if args.fold_id == -1:
        data_dir = os.path.join(data_dir, "fine")
    else:
        assert args.fold_id in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], (args.fold_id)
        data_dir = os.path.join(data_dir, "xval")
        data_dir = os.path.join(data_dir, "fold_{}".format(args.fold_id))
    args.data_dir = data_dir
    output_dir = os.path.join(args.output_dir, args.dataset)
    output_dir = os.path.join(output_dir, "adversarial")
    if args.fold_id > -1:
        output_dir = os.path.join(output_dir, "xval")
        output_dir = os.path.join(output_dir, "fold_{}".format(args.fold_id))
    else:
        output_dir = os.path.join(output_dir, "fine")
    train_data_file = os.path.join(data_dir, "train.json")
    dev_data_file = os.path.join(data_dir, "dev.json")
    test_data_file = os.path.join(data_dir, "test.json")
    label_list = labels_from_file(os.path.join(data_dir, args.label_file))
    label_level = int(args.label_file.split(".")[0].split("_")[-1])
    args.num_labels = len(label_list)
    args.label_level = label_level
    output_dir = os.path.join(output_dir, "l{}+{}".format(label_level, args.seed))
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir

    ## 2. define models
    args.model_name_or_path = os.path.join("data/pretrained_models", args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    config.HP_dropout = 0.5
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    model = AdversarialModelForRelCls(config=config, args=args)
    model = model.to(args.device)

    ## 3. prepare dataset
    dataset_params = {
        "relation_type": args.relation_type,
        "tokenizer": tokenizer,
        "max_seq_length": args.max_seq_length,
        "label_list": label_list,
        "label_level": label_level
    }

    if args.do_train:
        train_dataset = AdversarialDataset(train_data_file, params=dataset_params)
        dev_dataset = AdversarialDataset(dev_data_file, params=dataset_params)
        test_dataset = AdversarialDataset(test_data_file, params=dataset_params)
        print("Fold {} acc".format(args.fold_id))

        ## 1. init relation models
        print("############################################")
        print("######       Stage 1: Init Rel        ######")
        print("############################################")
        rel_output_dir = os.path.join(output_dir, "init_rel")
        args.rel_output_dir = rel_output_dir
        train_only_rel(model, args, train_dataset, dev_dataset, test_dataset, label_list, tokenizer, False)

        ## 2. init discriminator
        # fix the params in relation classification models
        print("############################################")
        print("######       Stage 2: Init Disc       ######")
        print("############################################")
        model.set_roberta_arg(do_fix=True)
        model.set_roberta_ori(do_fix=True)
        disc_output_dir = os.path.join(output_dir, "init_disc")
        args.disc_output_dir = disc_output_dir
        train_only_disc(model, args, train_dataset)
        # TIME_CHECKPOINT_DIR = ""

        ## 3. joint train models
        print("############################################")
        print("######      Stage 3: Joint Trian      ######")
        print("############################################")
        joint_output_dir = os.path.join(output_dir, "joint_train")
        args.joint_output_dir = joint_output_dir
        joint_train(model, args, train_dataset, dev_dataset, test_dataset, label_list, tokenizer)

    if args.do_dev or args.do_test:
        # """
        # l1_ji, 9, 5, 10, 10, 10
        seed_epoch = {106524: 9, 106464: 5, 106537: 10, 219539: 10, 430683: 10}
        epoch = seed_epoch[args.seed]
        checkpoint_file = os.path.join(args.output_dir, "joint_train/model/checkpoint_{}/pytorch_model.bin".format(epoch))
        model.load_state_dict(torch.load(checkpoint_file))
        args.output_dir = os.path.dirname(checkpoint_file)
        model.eval()

        if args.do_dev:
            dataset = AdversarialDataset(dev_data_file, params=dataset_params)
            acc, f1 = joint_evaluate(
                model, args, dataset, label_list, tokenizer, epoch, desc="dev", write_file=False
            )
            print(" Dev: acc=%.4f, f1=%.4f\n"%(acc, f1))
        if args.do_test:
            dataset = AdversarialDataset(test_data_file, params=dataset_params)
            acc, f1 = joint_evaluate(
                model, args, dataset, label_list, tokenizer, epoch, desc="test", write_file=False
            )
            print(" Test: acc=%.4f, f1=%.4f\n"%(acc, f1))
        # """
        """
        # dev_dataset = AdversarialDataset(dev_data_file, params=dataset_params)
        test_dataset = AdversarialDataset(test_data_file, params=dataset_params)
        join = os.path.join(args.output_dir, "joint_train/model/checkpoint_{}/pytorch_model.bin")
        temp_file = join
        for epoch in range(3, 11):
            checkpoint_file = temp_file.format(epoch)
            print(" Epoch %d, %s"%(epoch, checkpoint_file))
            model.load_state_dict(torch.load(checkpoint_file))
            model.eval()

            # acc, f1 = joint_evaluate(
            #     model, args, dev_dataset, label_list, tokenizer, epoch, desc="dev", write_file=True
            # )
            # print(" Dev: acc=%.4f, f1=%.4f" % (acc, f1))
            acc, f1 = joint_evaluate(
                model, args, test_dataset, label_list, tokenizer, epoch, desc="test", write_file=True
            )
            print(" Test: acc=%.4f, f1=%.4f" % (acc, f1))
            print()
        """

if __name__ == "__main__":
    main()
