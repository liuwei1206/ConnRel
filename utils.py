# author = liuwei
# date = 2022-04-11

import os
import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
import torch
import math
import random

random.seed(106524)

def cal_acc_f1_score_with_ids(pred_ids, label_ids, possible_label_ids):
    """
    sample_size: N
    label_size: V
    Args:
        pred_ids: [N]
        label_ids: [N]
        possible_label_ids: [N, V]
    note, each sample in implicit discourse may have more than one label
    if the predicted label match one of those labels, then the prediction is considered
    as true
    """
    extend_label_ids = []
    for idx, p in enumerate(pred_ids):
        if possible_label_ids[idx, p] == 1:
            extend_label_ids.append(p)
        else:
            extend_label_ids.append(label_ids[idx])
    label_ids = np.array(extend_label_ids)
    acc = accuracy_score(y_true=label_ids, y_pred=pred_ids)
    # p = precision_score(y_true=label_ids, y_pred=pred_ids, average="macro")
    # r = recall_score(y_true=label_ids, y_pred=pred_ids, average="macro")
    f1 = f1_score(y_true=label_ids, y_pred=pred_ids, average="macro")

    # return acc, p, r, f1
    return acc, f1

def cal_acc_f1_score_per_label(pred_ids, label_ids, possible_label_ids, label_list):
    """
    sample_size: N
    label_size: V
    Args:
        pred_ids: [N]
        label_ids: [N]
        possible_label_ids: [N, V]
    note, each sample in implicit discourse may have more than one label
    if the predicted label match one of those labels, then the prediction is considered
    as true
    """
    extend_label_ids = []
    for idx, p in enumerate(pred_ids):
        if possible_label_ids[idx, p] == 1:
            extend_label_ids.append(p)
        else:
            extend_label_ids.append(label_ids[idx])
    label_ids = np.array(extend_label_ids)
    res = classification_report(y_true=label_ids, y_pred=pred_ids, target_names=label_list, digits=4)
    print(res)

    return res

def count_frequency_in_files(file_names, rel_type="implicit", item_name="args1"):
    """
    Args:
        file_name:
        item_name: key name
    """
    item_frequency = {}
    total_num = 0
    for file in file_names:
        with open(file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    sample = json.loads(line)
                    if sample["relation_type"].lower() != rel_type:
                        continue
                    total_num += 1
                    item_value = sample[item_name].split("##")[0].split(".")[0]
                    item_value = item_value.lower()
                    if item_value in item_frequency:
                        item_frequency[item_value] += 1
                    else:
                        item_frequency[item_value] = 1

    return item_frequency, total_num

def get_connectives_with_threshold(data_dir, threshold=0.9):
    """
    Args:
        data_dir:
        threshold: if is a decimal, then reverse the top percent
                   if is a integer, then means the minimum frequency
    """
    # files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in ["train.json", "dev.json", "test.json"]]
    item_frequency, total_num = count_frequency_in_files(file_names=files, item_name="conn")
    conn_file = "connectives_with_threshold_{}.txt".format(threshold)
    conn_file = os.path.join(data_dir, conn_file)

    if os.path.exists(conn_file): # if exists, then read directly
        conns = []
        idfs = []
        with open(conn_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    items = line.split("\t")
                    conns.append(items[0].strip())
                    idfs.append(float(items[1].strip()))
        print("connectives number: ", len(conns))
        return conns, idfs

    conns = []
    frequencies = []
    new_total_num = 0
    if threshold > 1.0:
        for key in item_frequency.keys():
            if item_frequency[key] >= threshold:
                conns.append(key)
                frequencies.append(item_frequency[key])
                new_total_num += item_frequency[key]
    else:
        item_frequency = sorted(item_frequency.items(), key=lambda x: x[1], reverse=True)
        threshold_num = total_num * threshold
        cur_num = 0
        for item in item_frequency:
            conns.append(item[0])
            frequencies.append(int(item[1]))
            cur_num += item[1]
            new_total_num += item[1]
            if cur_num >= threshold_num:
                break

    if threshold == 1 or threshold == 1.0:
        pass
    else:
        conns.append("<unk>")
        frequencies.append(total_num - new_total_num)
    idfs = []
    with open(conn_file, "w", encoding="utf-8") as f:
        for con, fre in zip(conns, frequencies):
            idf = math.log(new_total_num / fre)
            # idf = 0.0
            idfs.append(idf)
            # print(idf)
            f.write("%s\t%.4f\n"%(con, idf))
    print("connectives number: ", len(conns))
    return conns, idfs

def get_onehot_conn_from_vocab(conns, tokenizer):
    """
    get the token_ids of each connective
    Args:
        conns:
        tokenizer
    """
    vocab_size = tokenizer.vocab_size
    conn_num = len(conns)
    conn_onehot_in_vocab = torch.zeros((conn_num, vocab_size)).float()
    conn_length_in_vocab = []

    for idx, conn in enumerate(conns):
        conn_tokens = tokenizer.tokenize(" " + conn.capitalize())
        # print(conn, " : ", conn_tokens)
        conn_length_in_vocab.append(len(conn_tokens))
        conn_token_ids = tokenizer.convert_tokens_to_ids(conn_tokens)
        conn_token_ids = torch.tensor(conn_token_ids).long()
        conn_onehot_in_vocab[idx, conn_token_ids] = 1

    return conn_onehot_in_vocab, torch.tensor(conn_length_in_vocab).float()

def split_train_for_pipeline_conn(file_name, train_percent=0.8, rel_type="implicit", save_res=True):
    """
    We split the train file (with annotated implicit connectives) of implicit relation file for
    step1 connective prediction

    Args:
        file_name: input train file for implicit relation classification
        train_percent: 80% for train,
        rel_type: implicit relation
        save_res: save the split or not
    """
    data_dir = os.path.dirname(file_name)
    train_file = "train_conns_for_pipeline_{}_percent_{}.json".format(rel_type, int(train_percent*100))
    dev_file = "dev_conns_for_pipeline_{}_percent_{}.json".format(rel_type, int((1-train_percent) * 100))
    train_file = os.path.join(data_dir, train_file)
    dev_file = os.path.join(data_dir, dev_file)

    if os.path.exists(train_file) and os.path.exists(dev_file):
        return train_file, dev_file

    # 1. filter samples with relation type
    total_filter_samples = []
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                sample = json.loads(line)
                relation_type = sample["relation_type"]
                if relation_type.upper() != rel_type.upper():
                    continue
                total_filter_samples.append(sample)

    # 2. split data
    random.shuffle(total_filter_samples)
    total_num = len(total_filter_samples)
    train_size = int(total_num * train_percent)
    train_samples = total_filter_samples[:train_size]
    dev_samples = total_filter_samples[train_size:]

    # 3. write to files
    with open(train_file, "w", encoding="utf-8") as f:
        for sample in train_samples:
            f.write("%s\n"%(json.dumps(sample, ensure_ascii=False)))

    with open(dev_file, "w", encoding="utf-8") as f:
        for sample in dev_samples:
            f.write("%s\n" % (json.dumps(sample, ensure_ascii=False)))

    return train_file, dev_file

def merge_pred_conn_to_file(file_name, all_predictions, rel_type="implicit"):
    """
    meger predicted connectives into original data file
    Args:
        file_name:
        all_predictions:
        rel_type:
    """
    dir_name = os.path.dirname(file_name)
    prefix_name = file_name.split("/")[-1].split(".")[0]
    new_file_name = "{}_with_pred_conn.json".format(prefix_name)
    new_file_name = os.path.join(dir_name, new_file_name)

    # 1. filter text
    total_filter_samples = []
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                sample = json.loads(line)
                relation_type = sample["relation_type"]
                if relation_type.upper() != rel_type.upper():
                    continue
                total_filter_samples.append(sample)

    # 2. align
    assert len(all_predictions) == len(total_filter_samples), (len(all_predictions), len(total_filter_samples))
    total_size = len(all_predictions)
    new_samples = []
    for idx in range(total_size):
        sample = total_filter_samples[idx]
        pred_conn = all_predictions[idx]
        sample["pred_conn"] = pred_conn
        new_samples.append(sample)

    # 3. write into file
    with open(new_file_name, "w", encoding="utf-8") as f:
        for sample in new_samples:
            f.write("%s\n"%(json.dumps(sample, ensure_ascii=False)))

    return new_file_name

def labels_from_file(label_file):
    label_list = []
    with open(label_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                label_list.append(line.strip().lower())

    return label_list
