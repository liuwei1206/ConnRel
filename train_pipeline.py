# author = liuwei
# date = 2022-06-05

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
from utils import cal_acc_f1_score_with_ids, get_connectives_with_threshold, get_onehot_conn_from_vocab
from utils import split_train_for_pipeline_conn, merge_pred_conn_to_file, labels_from_file

from task_dataset import ConnRobertaBaseDataset, RobertaBaseDataset
from models import RoBERTaForRelCls, RobertaForConnCls

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
    parser.add_argument("--pooling_type", default="mask", type=str, help="mask or cls")
    parser.add_argument("--label_file", default="labels_level_1.txt", type=str, help="the label file path")
    parser.add_argument("--use_conn", default=False, action="store_true")
    parser.add_argument("--conn_threshold", default=100.0, type=float)

    # for training
    parser.add_argument("--do_train", default=False, action="store_true")
    parser.add_argument("--do_dev", default=False, action="store_true")
    parser.add_argument("--do_test", default=False, action="store_true")
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--num_train_conn_epochs", default=10, type=int, help="training epoch for conn")
    parser.add_argument("--num_train_rel_epochs", default=10, type=int, help="training epoch for rel")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="learning rate")
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--warmup_ratio", default=0.06, type=float)
    parser.add_argument("--max_grad_norm", default=2.0, type=float)
    parser.add_argument("--seed", default=106524, type=int, help="random seed")

    return parser

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_dataloader(dataset, args, mode="train"):
    print("Dataset length: ", len(dataset))
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
    no_deday = ["bias", "LayerNorm.weigh"]
    specific_params = ["classifier"]
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

def train_conn(model, args, train_dataset, dev_dataset, conn_list, tokenizer):
    ## 1. prepare data
    train_dataloader = get_dataloader(train_dataset, args, mode="train")
    t_total = int(len(train_dataloader) * args.num_train_conn_epochs)
    num_train_epochs = args.num_train_conn_epochs
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
    best_dev = 0.0
    best_dev_epoch = 0
    res_list = []
    train_iterator = trange(1, int(num_train_epochs)+1, desc="Epoch")
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        model.train()
        model.zero_grad()
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "mask_position_ids": batch[2],
                "conn_ids": batch[3],
                "flag": "Train"
            }

            outputs = model(**inputs)
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
                print("current loss=%.4f, global average loss=%.4f"%(logging_loss, tr_loss / global_step))

        # evaluation and save
        model.eval()
        # train_acc, train_f1 = evaluate_conn(model, args, train_dataset, conn_list, tokenizer, epoch, desc="train")
        dev_acc, dev_f1 = evaluate_conn(model, args, dev_dataset, conn_list, tokenizer, epoch, desc="dev")
        res_list.append((dev_acc, dev_f1))
        print(" Epoch=%d"%(epoch))
        # print(" Train acc=%.4f, f1=%.4f"%(train_acc, train_f1))
        print(" Dev acc=%.4f, f1=%.4f" % (dev_acc, dev_f1))
        if dev_acc+dev_f1 > best_dev:
            best_dev = dev_acc + dev_f1
            best_dev_epoch = epoch

        output_dir = os.path.join(args.conn_output_dir, f"{PREFIX_CHECKPOINT_DIR}_{epoch}")
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    print(" Best dev: epoch=%d, acc=%.4f, f1=%.4f\n"%(
        best_dev_epoch, res_list[best_dev_epoch-1][0], res_list[best_dev_epoch-1][1]))
    with open(os.path.join(args.conn_output_dir, "best_epoch.txt"), "w", encoding="utf-8") as f:
        f.write("%s:%d\n"%("best_dev_epoch", best_dev_epoch))

    return best_dev_epoch

def evaluate_conn(model, args, dataset, conn_list, tokenizer, epoch, desc="dev", write_file=False,
                  evaluate_druing_train=True):
    if not evaluate_druing_train:
        desc = desc + "1" # guarantte a sequential reader
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
            "mask_position_ids": batch[2],
            "conn_ids": batch[3],
            "flag": "Eval"
        }

        with torch.no_grad():
            outputs = model(**inputs)
            preds = outputs[0]

        input_ids = batch[0].detach().cpu().numpy()
        label_ids = batch[3].detach().cpu().numpy()
        possible_label_ids = batch[4].detach().cpu().numpy()
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

    if evaluate_druing_train:
        if write_file:
            all_labels = [conn_list[int(idx)] for idx in all_label_ids]
            all_predictions = [conn_list[int(idx)] for idx in all_predict_ids]
            all_input_texts = [
                tokenizer.decode(all_input_ids[i], skip_special_tokens=True) for i in range(len(all_input_ids))
            ]
            output_dir = args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            file_name = "{}_res.txt".format(desc)
            file_name = os.path.join(output_dir, file_name)
            error_num = 0
            with open(file_name, "w", encoding="utf-8") as f:
                f.write("%-16s %-16s %s\n"%("Label", "Pred", "Text"))
                for label, pred, text in zip(all_labels, all_predictions, all_input_texts):
                    if label == pred:
                        f.write("%-16s %-16s %s\n"%(label, pred, text))
                    else:
                        error_num += 1
                        f.write("%-16s %-16s %s\n" % (label, pred, str(error_num) + " " + text))

        return acc, f1
    else:
        print(" %s: acc=%.4f, f1=%.4f" % (desc, acc, f1))
        all_predictions = [conn_list[int(idx)] for idx in all_predict_ids]
        return all_predictions

def train_rel(model, args, train_dataset, dev_dataset, test_dataset, label_list, tokenizer):
    ## 1. prepare data
    train_dataloader = get_dataloader(train_dataset, args, mode="train")
    t_total = int(len(train_dataloader) * args.num_train_rel_epochs)
    num_train_epochs = args.num_train_rel_epochs
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
    best_dev = 0.0
    best_dev_epoch = 0
    best_test = 0.0
    best_test_epoch = 0
    res_list = []
    train_iterator = trange(1, int(num_train_epochs)+1, desc="Epoch")
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        model.train()
        model.zero_grad()
        for step, batch in enumerate(epoch_iterator):
            batch_data = (batch[0], batch[1], batch[2])
            batch = tuple(t.to(args.device) for t in batch_data)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2],
                "flag": "Train"
            }

            outputs = model(**inputs)
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
                print("current loss=%.4f, global average loss=%.4f"%(logging_loss, tr_loss / global_step))

        # evaluation and save
        model.eval()
        # train_acc, train_f1 = evaluate_rel(model, args, train_dataset, label_list, tokenizer, epoch, desc="train")
        dev_acc, dev_f1 = evaluate_rel(model, args, dev_dataset, label_list, tokenizer, epoch, desc="dev")
        test_acc, test_f1 = evaluate_rel(model, args, test_dataset, label_list, tokenizer, epoch, desc="test")
        res_list.append((dev_acc, dev_f1, test_acc, test_f1))
        print(" Epoch=%d"%(epoch))
        # print(" Train acc=%.4f, f1=%.4f"%(epoch, train_acc, train_f1))
        print(" Dev acc=%.4f, f1=%.4f" % (dev_acc, dev_f1))
        print(" Test acc=%.4f, f1=%.4f"%(test_acc, test_f1))
        if dev_acc+dev_f1 > best_dev:
            best_dev = dev_acc + dev_f1
            best_dev_epoch = epoch
        if test_acc+test_f1 > best_test:
            best_test = test_acc + test_f1
            best_test_epoch = epoch

        # output_dir = os.path.join(args.rel_output_dir, TIME_CHECKPOINT_DIR)
        output_dir = os.path.join(args.rel_output_dir, "model")
        output_dir = os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}_{epoch}")
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    print(" Best dev: epoch=%d, acc=%.4f, f1=%.4f"%(
        best_dev_epoch, res_list[best_dev_epoch-1][0], res_list[best_dev_epoch-1][1])
    )
    print(" Best test: epoch=%d, acc=%.4f, f1=%.4f\n"%(
        best_test_epoch, res_list[best_test_epoch-1][2], res_list[best_test_epoch-1][3])
    )

def evaluate_rel(model, args, dataset, label_list, tokenizer, epoch, desc="dev", write_file=False):
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
            "labels": batch[2],
            "flag": "Eval"
        }

        with torch.no_grad():
            outputs = model(**inputs)
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
    if write_file:
        all_labels = [label_list[int(idx)] for idx in all_label_ids]
        all_predictions = [label_list[int(idx)] for idx in all_predict_ids]
        all_input_texts = [
            tokenizer.decode(all_input_ids[i], skip_special_tokens=True) for i in range(len(all_input_ids))
        ]
        file_name = os.path.join(args.data_dir, "pipe+{}_l{}+{}+{}.txt".format(
            desc, args.label_level, epoch, args.seed))
        error_num = 0
        with open(file_name, "w", encoding="utf-8") as f:
            f.write("%-16s %-16s %s\n"%("Label", "Pred", "Text"))
            for label, pred, text in zip(all_labels, all_predictions, all_input_texts):
                if label == pred:
                    f.write("%-16s %-16s %s\n"%(label, pred, text))
                else:
                    error_num += 1
                    f.write("%-16s %-16s %s\n" % (label, pred, str(error_num) + " " + text))

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

    ## 1. for conn prediction
    # 1.1 prepare data
    data_dir = os.path.join(args.data_dir, args.dataset)
    if args.fold_id == -1:
        data_dir = os.path.join(data_dir, "fine")
    else:
        assert args.fold_id in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], (args.fold_id)
        data_dir = os.path.join(data_dir, "xval")
        data_dir = os.path.join(data_dir, "fold_{}".format(args.fold_id))
    args.data_dir = data_dir
    output_dir = os.path.join(args.output_dir, args.dataset)
    output_dir = os.path.join(output_dir, "pipeline")
    if args.fold_id > -1:
        output_dir = os.path.join(output_dir, "xval")
        output_dir = os.path.join(output_dir, "fold_{}".format(args.fold_id))
    else:
        output_dir = os.path.join(output_dir, "fine")
    args.output_dir = output_dir
    conn_output_dir = os.path.join(output_dir, "conn+{}".format(args.seed))
    os.makedirs(conn_output_dir, exist_ok=True)
    args.conn_output_dir = conn_output_dir
    train_rel_file = os.path.join(data_dir, "train.json")
    train_conn_file, dev_conn_file = split_train_for_pipeline_conn(train_rel_file)
    conn_list, _ = get_connectives_with_threshold(args.data_dir, threshold=args.conn_threshold)
    args.num_connectives = len(conn_list)

    # judge whether the best connective prediction model already existed
    print("Fold {} acc".format(args.fold_id))
    best_epoch = -1
    if os.path.exists(os.path.join(conn_output_dir, "best_epoch.txt")):
        with open(os.path.join(conn_output_dir, "best_epoch.txt"), "r", encoding="utf-8") as f:
            line = f.readline()
            line = line.strip()
            items = line.split(":")
            if len(items) > 1:
                best_epoch = int(items[1])
            else:
                best_epoch = -1
    args.model_name_or_path = os.path.join("data/pretrained_models", args.model_name_or_path)
    if best_epoch < 0: # not existing checkpoints, we need to train from scratch
        # 1.2 define model for conn prediction
        config = RobertaConfig.from_pretrained(args.model_name_or_path)
        config.HP_dropout = 0.5
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
        conn_onehot_in_vocab, conn_length_in_vocab = get_onehot_conn_from_vocab(conn_list, tokenizer)
        args.conn_onehot_in_vocab = conn_onehot_in_vocab.to(args.device)
        args.conn_length_in_vocab = conn_length_in_vocab.to(args.device)
        model = RobertaForConnCls(config=config, args=args)
        model = model.to(args.device)

        # 1.3 prepare dataset
        dataset_params = {
            "relation_type": args.relation_type,
            "tokenizer": tokenizer,
            "max_seq_length": args.max_seq_length,
            "conn_list": conn_list,
            "pooling_type": args.pooling_type
        }

        # 1.4 train model
        train_dataset = ConnRobertaBaseDataset(train_conn_file, params=dataset_params)
        dev_dataset = ConnRobertaBaseDataset(dev_conn_file, params=dataset_params)
        best_epoch = train_conn(model, args, train_dataset, dev_dataset, conn_list, tokenizer)

    assert best_epoch > 0, (best_epoch)

    ## 2. init with the best checkpoint
    checkpoint_file = "{}_{}".format(PREFIX_CHECKPOINT_DIR, best_epoch)
    checkpoint_file = os.path.join(checkpoint_file, "pytorch_model.bin")
    checkpoint_file = os.path.join(conn_output_dir, checkpoint_file)

    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    config.HP_dropout = 0.5
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    conn_onehot_in_vocab, conn_length_in_vocab = get_onehot_conn_from_vocab(conn_list, tokenizer)
    args.conn_onehot_in_vocab = conn_onehot_in_vocab.to(args.device)
    args.conn_length_in_vocab = conn_length_in_vocab.to(args.device)
    conn_model = RobertaForConnCls(config=config, args=args)
    conn_model.load_state_dict(torch.load(checkpoint_file))
    conn_model = conn_model.to(args.device)

    ## 3. predict connectives for relation classification task
    # 3.1 dataset
    dev_rel_file = os.path.join(data_dir, "dev.json")
    test_rel_file = os.path.join(data_dir, "test.json")
    dataset_params = {
        "relation_type": args.relation_type,
        "tokenizer": tokenizer,
        "max_seq_length": args.max_seq_length,
        "conn_list": conn_list,
        "pooling_type": args.pooling_type
    }

    # 3.2 predict
    train_dataset = ConnRobertaBaseDataset(train_rel_file, params=dataset_params)
    dev_dataset = ConnRobertaBaseDataset(dev_rel_file, params=dataset_params)
    test_dataset = ConnRobertaBaseDataset(test_rel_file, params=dataset_params)
    all_train_predictions = evaluate_conn(
        conn_model, args, train_dataset, conn_list, tokenizer, best_epoch,
        desc="train", evaluate_druing_train=False
    )
    all_dev_predictions = evaluate_conn(
        conn_model, args, dev_dataset, conn_list, tokenizer, best_epoch,
        desc="dev", evaluate_druing_train=False
    )
    all_test_predictions = evaluate_conn(
        conn_model, args, test_dataset, conn_list, tokenizer, best_epoch,
        desc="test", evaluate_druing_train=False
    )

    # 3.3 merge to new file
    train_conn_rel_file = merge_pred_conn_to_file(train_rel_file, all_train_predictions)
    dev_conn_rel_file = merge_pred_conn_to_file(dev_rel_file, all_dev_predictions)
    test_conn_rel_file = merge_pred_conn_to_file(test_rel_file, all_test_predictions)

    ## 4. relation prediction
    # 4.1 data
    label_list = labels_from_file(os.path.join(data_dir, args.label_file))
    label_level = int(args.label_file.split(".")[0].split("_")[-1])
    args.label_level = label_level
    args.num_labels = len(label_list)
    rel_output_dir = os.path.join(output_dir, "rel")
    rel_output_dir = os.path.join(output_dir, "l{}+{}".format(label_level, args.seed))
    args.rel_output_dir = rel_output_dir
    os.makedirs(rel_output_dir, exist_ok=True)

    # 4.2 define model
    rel_model = RoBERTaForRelCls(config=config, args=args)
    rel_model = rel_model.to(args.device)

    # 4.3 dataset
    dataset_params = {
        "relation_type": args.relation_type,
        "tokenizer": tokenizer,
        "max_seq_length": args.max_seq_length,
        "label_list": label_list,
        "label_level": label_level,
        "use_conn": args.use_conn,
        "conn_type": "predict"
    }

    if args.do_train:
        train_dataset = RobertaBaseDataset(train_conn_rel_file, params=dataset_params)
        dev_dataset = RobertaBaseDataset(dev_conn_rel_file, params=dataset_params)
        test_dataset = RobertaBaseDataset(test_conn_rel_file, params=dataset_params)
        train_rel(rel_model, args, train_dataset, dev_dataset, test_dataset, label_list, tokenizer)

    if args.do_dev or args.do_test:
        checkpoint_file = ""
        epoch = 0
        rel_model.load_state_dict(torch.load(checkpoint_file))
        args.output_dir = os.path.dirname(checkpoint_file)
        rel_model.eval()

        # dataset = RobertaBaseDataset(train_data_file, params=dataset_params)
        # acc, f1 = evaluate_rel(
        #     rel_model, args, dataset, label_list, tokenizer,
        #     epoch, desc="train", write_file=True
        # )
        # print(" Train: acc=%.4f, f1=%.4f\n" % (acc, f1))

        if args.do_dev:
            dataset = RobertaBaseDataset(dev_data_file, params=dataset_params)
            acc, f1 = evaluate_rel(
                rel_model, args, dataset, label_list, tokenizer,
                epoch, desc="dev", write_file=False)
            print("Dev: acc=%.4f, f1=%.4f\n" % (acc, f1))
        if args.do_test:
            dataset = RobertaBaseDataset(test_data_file, params=dataset_params)
            acc, f1 = evaluate_rel(
                rel_model, args, dataset, label_list, tokenizer,
                epoch, desc="test", write_file=False
            )
            print("Test: acc=%.4f, f1=%.4f\n" % (acc, f1))

if __name__ == "__main__":
    main()
