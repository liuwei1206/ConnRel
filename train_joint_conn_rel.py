# author = liuwei
# date = 2022-04-15

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
from utils import cal_acc_f1_score_with_ids, get_connectives_with_threshold, get_onehot_conn_from_vocab, labels_from_file
from task_dataset import JointRobertaBaseDataset
from models import JointConnRel

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
    parser.add_argument("--fold_id", default=-1, type=int, help="for normal split, we set it as -1; for xval, ")

    # hyperparameters
    parser.add_argument("--relation_type", default="implicit", type=str)
    parser.add_argument("--label_file", default="labels_level_1.txt", type=str, help="the label file path")
    parser.add_argument("--conn_threshold", default=100.0, type=float)
    parser.add_argument("--sample_k", default=100, type=int, help="100 for pdtb2, 200 for pdtb3")

    # for training
    parser.add_argument("--do_train", default=False, action="store_true")
    parser.add_argument("--do_dev", default=False, action="store_true")
    parser.add_argument("--do_test", default=False, action="store_true")
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--num_train_epochs", default=10, type=int, help="training epoch")
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

def train(model, args, train_dataset, dev_dataset, test_dataset, conn_list, label_list, tokenizer):
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
    tr_loss = 0.0
    logging_loss = 0.0
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
            global_step += 1
            if args.sample_k > 0:
                sample_probability = args.sample_k / (args.sample_k + math.exp(global_step / args.sample_k))
            else:
                sample_probability = -1.0

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "mask_position_ids": batch[2],
                "sample_p": sample_probability,
                "conn_ids": batch[3],
                "labels": batch[4],
                "flag": "Train"
            }

            outputs = model(**inputs)
            loss = outputs[0]
            # loss = outputs[2]
            conn_loss = outputs[1]
            rel_loss = outputs[2]

            optimizer.zero_grad()
            loss.backward()
            logging_loss = loss.item()
            tr_loss += logging_loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            if global_step % print_step == 0:
                print("current conn_loss=%.4f, rel_loss=%.4f, loss=%.4f, global average loss=%.4f" % (
                conn_loss.item(), rel_loss.item(), logging_loss, tr_loss / global_step))

        # evaluation and save
        model.eval()
        # train_conn_acc, train_acc, train_f1 = evaluate(
        #     model, args, train_dataset, conn_list, label_list, tokenizer, epoch, desc="train"
        # )
        dev_conn_acc, dev_acc, dev_f1 = evaluate(
            model, args, dev_dataset, conn_list, label_list, tokenizer, epoch, desc="dev"
        )
        test_conn_acc, test_acc, test_f1 = evaluate(
            model, args, test_dataset, conn_list, label_list, tokenizer, epoch, desc="test"
        )
        res_list.append((dev_acc, dev_f1, test_acc, test_f1))
        print(" Epoch=%d"%(epoch))
        # print(" Train conn_acc=%.4f, acc=%.4f, f1=%.4f"%(epoch, train_conn_acc, train_acc, train_f1))
        print(" Dev conn_acc=%.4f, acc=%.4f, f1=%.4f"%(dev_conn_acc, dev_acc, dev_f1))
        print(" Test conn_acc=%.4f, acc=%.4f, f1=%.4f"%(test_conn_acc, test_acc, test_f1))
        if dev_acc+dev_f1 > best_dev:
            best_dev = dev_acc + dev_f1
            best_dev_epoch = epoch
        if test_acc+test_f1 > best_test:
            best_test = test_acc + test_f1
            best_test_epoch = epoch

        # output_dir = os.path.join(args.output_dir, TIME_CHECKPOINT_DIR)
        output_dir = os.path.join(args.output_dir, "model")
        output_dir = os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}_{epoch}")
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    print(" Best dev: epoch=%d, acc=%.4f, f1=%.4f"%(
        best_dev_epoch, res_list[best_dev_epoch-1][0], res_list[best_dev_epoch-1][1])
    )
    print(" Best test: epoch=%d, acc=%.4f, f1=%.4f\n"%(
        best_test_epoch, res_list[best_test_epoch-1][2], res_list[best_test_epoch-1][3])
    )

def evaluate(model, args, dataset, conn_list, label_list, tokenizer, epoch, desc="dev", write_file=False):
    dataloader = get_dataloader(dataset, args, mode=desc)

    all_input_ids = None
    all_conn_ids = None
    all_pred_conn_ids = None
    all_label_ids = None
    all_possible_label_ids = None
    all_predict_ids = None
    for batch in tqdm(dataloader, desc=desc):
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "mask_position_ids": batch[2],
            "conn_ids": batch[3],
            "labels": batch[4],
            "flag": "Eval"
        }
        with torch.no_grad():
            outputs = model(**inputs)
            conn_preds = outputs[0]
            rel_preds = outputs[1]

        input_ids = batch[0].detach().cpu().numpy()
        conn_ids = batch[3].detach().cpu().numpy()
        label_ids = batch[4].detach().cpu().numpy()
        possible_label_ids = batch[5].detach().cpu().numpy()
        # print(possible_label_ids)
        pred_conn_ids = conn_preds.detach().cpu().numpy()
        pred_ids = rel_preds.detach().cpu().numpy()
        if all_label_ids is None:
            all_input_ids = input_ids
            all_conn_ids = conn_ids
            all_pred_conn_ids = pred_conn_ids
            all_label_ids = label_ids
            all_possible_label_ids = possible_label_ids
            all_predict_ids = pred_ids
        else:
            all_input_ids = np.append(all_input_ids, input_ids, axis=0)
            all_conn_ids = np.append(all_conn_ids, conn_ids)
            all_pred_conn_ids = np.append(all_pred_conn_ids, pred_conn_ids)
            all_label_ids = np.append(all_label_ids, label_ids)
            all_possible_label_ids = np.append(all_possible_label_ids, possible_label_ids, axis=0)
            all_predict_ids = np.append(all_predict_ids, pred_ids)

    conn_acc = np.sum(all_conn_ids == all_pred_conn_ids) / all_conn_ids.shape[0]
    acc, f1 = cal_acc_f1_score_with_ids(
        pred_ids=all_predict_ids,
        label_ids=all_label_ids,
        possible_label_ids=all_possible_label_ids
    )

    if write_file:
        all_conns = [conn_list[int(idx)] for idx in all_conn_ids]
        all_pred_conns = [conn_list[int(idx)] for idx in all_pred_conn_ids]
        all_labels = [label_list[int(idx)] for idx in all_label_ids]
        all_predictions = [label_list[int(idx)] for idx in all_predict_ids]
        all_input_texts = [
            tokenizer.decode(all_input_ids[i], skip_special_tokens=True) for i in range(len(all_input_ids))
        ]

        file_name = os.path.join(args.data_dir, "joint+{}_l{}+{}+{}.txt".format(
            desc, args.label_level, epoch, args.seed))
        error_num = 0
        with open(file_name, "w", encoding="utf-8") as f:
            f.write("%-16s %-16s %-16s %-16s %s\n" % ("Conn", "Pred_conn", "Label", "Pred", "Text"))
            for conn, pred_conn, label, pred, text in zip(
                    all_conns, all_pred_conns, all_labels, all_predictions, all_input_texts
            ):
                if label == pred:
                    f.write("%-16s %-16s %-16s %-16s %s\n" % (conn, pred_conn, label, pred, text))
                else:
                    error_num += 1
                    f.write("%-16s %-16s %-16s %-16s %s\n" % (conn, pred_conn, label, pred, str(error_num) + " " + text))

    return conn_acc, acc, f1

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
        data_dir = os.path.join(data_dir, "xval")
        data_dir = os.path.join(data_dir, "fold_{}".format(args.fold_id))
    args.data_dir = data_dir
    output_dir = os.path.join(args.output_dir, args.dataset)
    output_dir = os.path.join(output_dir, "joint_conn_rel")
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
    output_dir = os.path.join(output_dir, "l{}+{}".format(label_level,args.seed))
    os.makedirs(output_dir, exist_ok=True)
    args.label_level = label_level
    args.output_dir = output_dir
    conn_list, _ = get_connectives_with_threshold(args.data_dir, threshold=args.conn_threshold)
    args.num_labels = len(label_list)
    args.num_connectives = len(conn_list)

    ## 2. define models
    args.model_name_or_path = os.path.join("data/pretrained_models", args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    config.HP_dropout = args.dropout
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    conn_onehot_in_vocab, conn_length_in_vocab = get_onehot_conn_from_vocab(conn_list, tokenizer)
    args.conn_onehot_in_vocab = conn_onehot_in_vocab.to(args.device)
    args.conn_length_in_vocab = conn_length_in_vocab.to(args.device)
    model = JointConnRel(config=config, args=args)
    model = model.to(args.device)

    ## 3. prepare dataset
    dataset_params = {
        "relation_type": args.relation_type,
        "tokenizer": tokenizer,
        "max_seq_length": args.max_seq_length,
        "label_list": label_list,
        "label_level": label_level,
        "connective_list": conn_list
    }

    if args.do_train:
        train_dataset = JointRobertaBaseDataset(train_data_file, params=dataset_params)
        dev_dataset = JointRobertaBaseDataset(dev_data_file, params=dataset_params)
        test_dataset = JointRobertaBaseDataset(test_data_file, params=dataset_params)
        print("Fold {} acc".format(args.fold_id))
        train(model, args, train_dataset, dev_dataset, test_dataset, conn_list, label_list, tokenizer)

    if args.do_dev or args.do_test:
        """
        check_dir = os.path.join(args.output_dir, "model")
        # l1_ji, 5, 8, 9, 5, 9
        # l2_ji
        seed_epoch = {106524: 5, 106464: 8, 106537: 9, 219539: 5, 430683: 7}
        epoch = seed_epoch[args.seed]
        checkpoint_file = os.path.join(check_dir, "checkpoint_{}/pytorch_model.bin".format(epoch))
        print(checkpoint_file)
        args.output_dir = os.path.dirname(checkpoint_file)
        model.load_state_dict(torch.load(checkpoint_file))
        model.eval()
        
        # dataset = JointRobertaBaseDataset(train_data_file, params=dataset_params)
        # conn_acc, acc, f1 = evaluate(
        #     model, args, dataset, conn_list, label_list, tokenizer, 
        #     epoch, desc="train", write_file=False
        # )
        
        print("Train: conn_acc=%.4f, acc=%.4f, f1=%.4f\n"%(conn_acc, acc, f1))
        if args.do_dev:
            dataset = JointRobertaBaseDataset(dev_data_file, params=dataset_params)
            conn_acc, acc, f1 = evaluate(
                model, args, dataset, conn_list, label_list, tokenizer,
                epoch, desc="dev", write_file=False
            )
            print(" Dev: conn_acc=%.4f, acc=%.4f, f1=%.4f\n" % (conn_acc, acc, f1))
        if args.do_test:
            dataset = JointRobertaBaseDataset(test_data_file, params=dataset_params)
            conn_acc, acc, f1 = evaluate(
                model, args, dataset, conn_list, label_list, tokenizer,
                epoch, desc="test", write_file=False
            )
            print("Test: conn_acc=%.4f, acc=%.4f, f1=%.4f\n" % (conn_acc, acc, f1))
        """
        # dev_dataset = JointRobertaBaseDataset(dev_data_file, params=dataset_params)
        test_dataset = JointRobertaBaseDataset(test_data_file, params=dataset_params)
        join = os.path.join(args.output_dir, "model/checkpoint_{}/pytorch_model.bin")
        temp_file = join
        for epoch in range(3, 11):
            checkpoint_file = temp_file.format(epoch)
            print(" Epoch %d, %s"%(epoch, checkpoint_file))
            model.load_state_dict(torch.load(checkpoint_file))
            model.eval()

            # conn_acc, acc, f1 = evaluate(
            #     model, args, dev_dataset, conn_list, label_list, tokenizer,
            #     epoch, desc="dev", write_file=False
            # )
            # print(" Dev: conn_acc=%.4f, acc=%.4f, f1=%.4f" % (conn_acc, acc, f1))
            conn_acc, acc, f1 = evaluate(
                model, args, test_dataset, conn_list, label_list, tokenizer,
                epoch, desc="test", write_file=False
            )
            print(" Test: conn_acc=%.4f, acc=%.4f, f1=%.4f" % (conn_acc, acc, f1))
            print()

if __name__ == "__main__":
    main()
