# author = liuwei
# date = 2022-04-11

import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset

class RobertaBaseDataset(Dataset):
    """
    Args:
        file_name: the input file data
        params:
            relation_type: explicit or implicit
            tokenizer:
            max_seq_length
    """
    def __init__(self, file_name, params):
        self.input_file_name = file_name
        self.relation_type = params["relation_type"]
        self.tokenizer = params["tokenizer"]
        self.max_seq_length = params["max_seq_length"]
        self.label_list = params["label_list"]
        self.label_level = params["label_level"] - 1 # start from 0
        self.use_conn = params["use_conn"]
        if "conn_type" in params.keys():
            self.conn_type = params["conn_type"]
        else:
            self.conn_type = "ground"

        self.init_np_dataset()

    def init_np_dataset(self):
        all_input_ids = []
        all_attention_mask = []
        all_label_ids = []
        all_possible_label_ids = []

        with open(self.input_file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    sample = json.loads(line)
                    relation_type = sample["relation_type"]
                    if relation_type.upper() != self.relation_type.upper():
                        continue

                    arg1 = sample["arg1"]
                    arg2 = sample["arg2"]
                    all_level_relation_class = sample["relation_class"].split("##")[0].split(".")
                    possible_all_level_relation_classes = [item.split(".") for item in sample["relation_class"].split("##")]
                    if len(all_level_relation_class) > self.label_level:
                        relation_class = all_level_relation_class[self.label_level].lower()
                    else:
                        relation_class = None
                    possible_relation_classes = [item[self.label_level].lower() for item in possible_all_level_relation_classes if len(item) > self.label_level]
                    possible_relation_classes = [item for item in possible_relation_classes if item in self.label_list]
                    connectives = sample["conn"].split("##")[0].lower()

                    if (relation_class is None) or (relation_class not in self.label_list):
                        continue

                    if self.use_conn:
                        if self.conn_type.lower() == "ground":
                            connective = sample["conn"].split("##")[0].strip()
                        elif self.conn_type.lower() == "predict":
                            connective = sample["pred_conn"].strip()
                        else:
                            connective = ""
                        connective = connective.capitalize() + " "
                    else:
                        connective = ""

                    # for encoder input_ids
                    tokens_1 = self.tokenizer.tokenize(arg1)
                    tokens_2 = self.tokenizer.tokenize(connective + arg2)
                    tokens = ["<s>"] + tokens_1 + ["</s>", "</s>"] + tokens_2
                    if len(tokens) > self.max_seq_length - 1:
                        tokens = tokens[:self.max_seq_length - 1]
                    tokens = tokens + ["</s>"]
                    token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                    label_id = self.label_list.index(relation_class)
                    possible_label_ids = np.zeros(len(self.label_list), dtype=np.int)
                    for label in possible_relation_classes:
                        possible_label_ids[self.label_list.index(label)] = 1

                    # padding
                    input_ids = np.ones(self.max_seq_length, dtype=np.int)
                    attention_mask = np.zeros(self.max_seq_length, dtype=np.int)
                    input_ids = input_ids * self.tokenizer.pad_token_id
                    input_ids[:len(token_ids)] = token_ids
                    attention_mask[:len(token_ids)] = 1

                    all_input_ids.append(input_ids)
                    all_attention_mask.append(attention_mask)
                    all_label_ids.append(label_id)
                    all_possible_label_ids.append(possible_label_ids)

        assert len(all_input_ids) == len(all_attention_mask), (len(all_input_ids), len(all_attention_mask))
        assert len(all_input_ids) == len(all_label_ids), (len(all_input_ids), len(all_label_ids))
        assert len(all_input_ids) == len(all_possible_label_ids), (len(all_input_ids), len(all_possible_label_ids))

        all_input_ids = np.array(all_input_ids)
        all_attention_mask = np.array(all_attention_mask)
        all_label_ids = np.array(all_label_ids)
        all_possible_label_ids = np.array(all_possible_label_ids)

        self.input_ids = all_input_ids
        self.attention_mask = all_attention_mask
        self.label_ids = all_label_ids
        self.possible_label_ids = all_possible_label_ids

        self.total_size = self.input_ids.shape[0]

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        return (
            torch.tensor(self.input_ids[index]),
            torch.tensor(self.attention_mask[index]),
            torch.tensor(self.label_ids[index]),
            torch.tensor(self.possible_label_ids[index])
        )


class ConnRobertaBaseDataset(Dataset):
    """
    Args:
        file_name: the input file data
        params:
            relation_type: explicit or implicit
            tokenizer:
            max_seq_length
    """
    def __init__(self, file_name, params):
        self.input_file_name = file_name
        self.relation_type = params["relation_type"]
        self.tokenizer = params["tokenizer"]
        self.max_seq_length = params["max_seq_length"]
        self.conn_list = params["conn_list"]
        self.pooling_type = params["pooling_type"]

        self.init_np_dataset()

    def init_np_dataset(self):
        all_input_ids = []
        all_attention_mask = []
        all_mask_position_ids = []
        all_conn_ids = []
        all_possible_conn_ids = []

        with open(self.input_file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    sample = json.loads(line)
                    relation_type = sample["relation_type"]
                    if relation_type.upper() != self.relation_type.upper():
                        continue

                    arg1 = sample["arg1"].replace("\n", ". ").strip()
                    arg2 = sample["arg2"].replace("\n", ". ").strip()
                    connectives = sample["conn"].split("##")[0].lower()
                    possible_connectives = [item.strip().lower() for item in sample["conn"].split("##")]

                    # for encoder input_ids
                    tokens_1 = self.tokenizer.tokenize(arg1)
                    tokens_2 = self.tokenizer.tokenize(arg2)
                    if self.pooling_type == "cls":
                        tokens = ["<s>"] + tokens_1 + ["</s>", "</s>"] + tokens_2
                    else:
                        tokens = ["<s>"] + tokens_1 + ["<mask>"] + tokens_2
                    if len(tokens) > self.max_seq_length - 1:
                        tokens = tokens[:self.max_seq_length - 1]
                    tokens = tokens + ["</s>"]
                    token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                    mask_position_id = len(tokens_1) + 1
                    assert mask_position_id < self.max_seq_length, (mask_position_id, self.max_seq_length)

                    if connectives in self.conn_list:
                        conn_id = self.conn_list.index(connectives)
                    else:
                        conn_id = self.conn_list.index("<unk>")
                    possible_conn_ids = np.zeros(len(self.conn_list), dtype=np.int)
                    for conn in possible_connectives:
                        if conn in self.conn_list:
                            tmp_idx = self.conn_list.index(conn)
                        else:
                            tmp_idx = self.conn_list.index("<unk>")
                        possible_conn_ids[tmp_idx] = 1

                    # padding
                    input_ids = np.ones(self.max_seq_length, dtype=np.int)
                    attention_mask = np.zeros(self.max_seq_length, dtype=np.int)
                    input_ids = input_ids * self.tokenizer.pad_token_id
                    input_ids[:len(token_ids)] = token_ids
                    attention_mask[:len(token_ids)] = 1

                    all_input_ids.append(input_ids)
                    all_attention_mask.append(attention_mask)
                    all_mask_position_ids.append(mask_position_id)
                    all_conn_ids.append(conn_id)
                    all_possible_conn_ids.append(possible_conn_ids)

        assert len(all_input_ids) == len(all_attention_mask), (len(all_input_ids), len(all_attention_mask))
        assert len(all_input_ids) == len(all_mask_position_ids), (len(all_input_ids), len(all_mask_position_ids))
        assert len(all_input_ids) == len(all_conn_ids), (len(all_input_ids), len(all_conn_ids))
        assert len(all_input_ids) == len(all_possible_conn_ids), (len(all_input_ids), len(all_possible_conn_ids))

        all_input_ids = np.array(all_input_ids)
        all_attention_mask = np.array(all_attention_mask)
        all_mask_position_ids = np.array(all_mask_position_ids)
        all_conn_ids = np.array(all_conn_ids)
        all_possible_conn_ids = np.array(all_possible_conn_ids)

        self.input_ids = all_input_ids
        self.attention_mask = all_attention_mask
        self.mask_position_ids = all_mask_position_ids
        self.conn_ids = all_conn_ids
        self.possible_conn_ids = all_possible_conn_ids

        self.total_size = self.input_ids.shape[0]

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        return (
            torch.tensor(self.input_ids[index]),
            torch.tensor(self.attention_mask[index]),
            torch.tensor(self.mask_position_ids[index]),
            torch.tensor(self.conn_ids[index]),
            torch.tensor(self.possible_conn_ids[index])
        )


class JointRobertaBaseDataset(Dataset):
    """
    Args:
        file_name: the input file data
        params:
            relation_type: explicit or implicit
            tokenizer:
            max_seq_length
    """
    def __init__(self, file_name, params):
        self.input_file_name = file_name
        self.relation_type = params["relation_type"]
        self.tokenizer = params["tokenizer"]
        self.max_seq_length = params["max_seq_length"]
        self.label_list = params["label_list"]
        self.label_level = params["label_level"] - 1 # start from 0
        self.connective_list = params["connective_list"]

        self.init_np_dataset()

    def init_np_dataset(self):
        all_input_ids = []
        all_attention_mask = []
        all_mask_position_ids = []
        all_conn_ids = []
        all_label_ids = []
        all_possible_label_ids = [] # each sample can have multi_labels

        with open(self.input_file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    sample = json.loads(line)
                    relation_type = sample["relation_type"]
                    if relation_type.upper() != self.relation_type.upper():
                        continue

                    arg1 = sample["arg1"].replace("\n", ". ").strip()
                    arg2 = sample["arg2"].replace("\n", ". ").strip()
                    all_level_relation_class = sample["relation_class"].split("##")[0].split(".")
                    possible_all_level_relation_classes = [item.split(".") for item in sample["relation_class"].split("##")]
                    if len(all_level_relation_class) > self.label_level:
                        relation_class = all_level_relation_class[self.label_level].lower()
                    else:
                        relation_class = None
                    possible_relation_classes = [item[self.label_level].lower() for item in possible_all_level_relation_classes if len(item) > self.label_level]
                    possible_relation_classes = [item for item in possible_relation_classes if item in self.label_list]
                    connectives = sample["conn"].split("##")[0].lower()

                    if (relation_class is None) or (relation_class not in self.label_list):
                        continue

                    # for encoder input_ids
                    tokens_1 = self.tokenizer.tokenize(arg1)
                    tokens_2 = self.tokenizer.tokenize(arg2)
                    # tokens = ["<s>"] + tokens_1 + ["<mask>"] + tokens_2
                    tokens = ["<s>"] + tokens_1 + self.tokenizer.tokenize(connectives) + tokens_2
                    if len(tokens) > self.max_seq_length - 1:
                        tokens = tokens[:self.max_seq_length - 1]
                    tokens = tokens + ["</s>"]
                    token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                    mask_position_id = len(tokens_1) + 1
                    assert mask_position_id < self.max_seq_length, (mask_position_id, self.max_seq_length)
                    if connectives in self.connective_list:
                        conn_id = self.connective_list.index(connectives)
                    else:
                        conn_id = self.connective_list.index("<unk>")
                    label_id = self.label_list.index(relation_class)
                    possible_label_ids = np.zeros(len(self.label_list), dtype=np.int)
                    for label in possible_relation_classes:
                        possible_label_ids[self.label_list.index(label)] = 1

                    # padding
                    input_ids = np.ones(self.max_seq_length, dtype=np.int)
                    attention_mask = np.zeros(self.max_seq_length, dtype=np.int)
                    input_ids = input_ids * self.tokenizer.pad_token_id
                    input_ids[:len(token_ids)] = token_ids
                    attention_mask[:len(token_ids)] = 1

                    all_input_ids.append(input_ids)
                    all_attention_mask.append(attention_mask)
                    all_mask_position_ids.append(mask_position_id)
                    all_conn_ids.append(conn_id)
                    all_label_ids.append(label_id)
                    all_possible_label_ids.append(possible_label_ids)

        assert len(all_input_ids) == len(all_attention_mask), (len(all_input_ids), len(all_attention_mask))
        assert len(all_input_ids) == len(all_mask_position_ids), (len(all_input_ids), len(all_mask_position_ids))
        assert len(all_input_ids) == len(all_conn_ids), (len(all_input_ids), len(all_conn_ids))
        assert len(all_input_ids) == len(all_label_ids), (len(all_input_ids), len(all_label_ids))
        assert len(all_input_ids) == len(all_possible_label_ids), (len(all_input_ids), len(all_possible_label_ids))

        all_input_ids = np.array(all_input_ids)
        all_attention_mask = np.array(all_attention_mask)
        all_mask_position_ids = np.array(all_mask_position_ids)
        all_conn_ids = np.array(all_conn_ids)
        all_label_ids = np.array(all_label_ids)
        all_possible_label_ids = np.array(all_possible_label_ids)

        self.input_ids = all_input_ids
        self.attention_mask = all_attention_mask
        self.mask_position_ids = all_mask_position_ids
        self.conn_ids = all_conn_ids
        self.label_ids = all_label_ids
        self.possible_label_ids = all_possible_label_ids

        self.total_size = self.input_ids.shape[0]

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        return (
            torch.tensor(self.input_ids[index]),
            torch.tensor(self.attention_mask[index]),
            torch.tensor(self.mask_position_ids[index]),
            torch.tensor(self.conn_ids[index]),
            torch.tensor(self.label_ids[index]),
            torch.tensor(self.possible_label_ids[index])
        )


class MultiTaskDataset(Dataset):
    def __init__(self, file_name, params):
        self.input_file_name = file_name
        self.relation_type = params["relation_type"]
        self.tokenizer = params["tokenizer"]
        self.max_seq_length = params["max_seq_length"]
        self.label_list = params["label_list"]
        self.label_level = params["label_level"] - 1 # start from 0
        self.connective_list = params["connective_list"]

        self.init_np_dataset()

    def init_np_dataset(self):
        all_input_ids = []
        all_attention_mask = []
        all_conn_ids = []
        all_label_ids = []
        all_possible_label_ids = [] # each sample can have multi_labels

        with open(self.input_file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    sample = json.loads(line)
                    relation_type = sample["relation_type"]
                    if relation_type.upper() != self.relation_type.upper():
                        continue

                    arg1 = sample["arg1"].replace("\n", ". ").strip()
                    arg2 = sample["arg2"].replace("\n", ". ").strip()
                    all_level_relation_class = sample["relation_class"].split("##")[0].split(".")
                    possible_all_level_relation_classes = [item.split(".") for item in sample["relation_class"].split("##")]
                    if len(all_level_relation_class) > self.label_level:
                        relation_class = all_level_relation_class[self.label_level].lower()
                    else:
                        relation_class = None
                    possible_relation_classes = [item[self.label_level].lower() for item in possible_all_level_relation_classes if len(item) > self.label_level]
                    possible_relation_classes = [item for item in possible_relation_classes if item in self.label_list]
                    connectives = sample["conn"].split("##")[0].lower()

                    if (relation_class is None) or (relation_class not in self.label_list):
                        continue

                    # for encoder input_ids
                    tokens_1 = self.tokenizer.tokenize(arg1)
                    tokens_2 = self.tokenizer.tokenize(arg2)
                    tokens = ["<s>"] + tokens_1 + ["</s>", "</s>"] + tokens_2
                    if len(tokens) > self.max_seq_length - 1:
                        tokens = tokens[:self.max_seq_length - 1]
                    tokens = tokens + ["</s>"]
                    token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                    if connectives in self.connective_list:
                        conn_id = self.connective_list.index(connectives)
                    else:
                        conn_id = self.connective_list.index("<unk>")
                    label_ids = self.label_list.index(relation_class)
                    possible_label_ids = np.zeros(len(self.label_list), dtype=np.int)
                    for label in possible_relation_classes:
                        possible_label_ids[self.label_list.index(label)] = 1

                    # padding
                    input_ids = np.ones(self.max_seq_length, dtype=np.int)
                    attention_mask = np.zeros(self.max_seq_length, dtype=np.int)
                    input_ids = input_ids * self.tokenizer.pad_token_id
                    input_ids[:len(token_ids)] = token_ids
                    attention_mask[:len(token_ids)] = 1

                    all_input_ids.append(input_ids)
                    all_attention_mask.append(attention_mask)
                    all_conn_ids.append(conn_id)
                    all_label_ids.append(label_ids)
                    all_possible_label_ids.append(possible_label_ids)

        assert len(all_input_ids) == len(all_attention_mask), (len(all_input_ids), len(all_attention_mask))
        assert len(all_input_ids) == len(all_conn_ids), (len(all_input_ids), len(all_conn_ids))
        assert len(all_input_ids) == len(all_label_ids), (len(all_input_ids), len(all_label_ids))
        assert len(all_input_ids) == len(all_possible_label_ids), (len(all_input_ids), len(all_possible_label_ids))

        all_input_ids = np.array(all_input_ids)
        all_attention_mask = np.array(all_attention_mask)
        all_conn_ids = np.array(all_conn_ids)
        all_label_ids = np.array(all_label_ids)
        all_possible_label_ids = np.array(all_possible_label_ids)

        self.input_ids = all_input_ids
        self.attention_mask = all_attention_mask
        self.conn_ids = all_conn_ids
        self.label_ids = all_label_ids
        self.possible_label_ids = all_possible_label_ids

        self.total_size = self.input_ids.shape[0]

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        return (
            torch.tensor(self.input_ids[index]),
            torch.tensor(self.attention_mask[index]),
            torch.tensor(self.conn_ids[index]),
            torch.tensor(self.label_ids[index]),
            torch.tensor(self.possible_label_ids[index])
        )


class AdversarialDataset(Dataset):
    def __init__(self, file_name, params):
        self.input_file_name = file_name
        self.relation_type = params["relation_type"]
        self.tokenizer = params["tokenizer"]
        self.max_seq_length = params["max_seq_length"]
        self.label_list = params["label_list"]
        self.label_level = params["label_level"] - 1  # start from 0

        self.init_np_dataset()

    def init_np_dataset(self):
        all_input_ids = []
        all_attention_mask = []
        all_arg_input_ids = []
        all_arg_attention_mask = []
        all_label_ids = []
        all_possible_label_ids = []  # each sample can have multi_labels

        with open(self.input_file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    sample = json.loads(line)
                    relation_type = sample["relation_type"]
                    if relation_type.upper() != self.relation_type.upper():
                        continue

                    arg1 = sample["arg1"].replace("\n", ". ").strip()
                    arg2 = sample["arg2"].replace("\n", ". ").strip()
                    all_level_relation_class = sample["relation_class"].split("##")[0].split(".")
                    possible_all_level_relation_classes = [item.split(".") for item in sample["relation_class"].split("##")]
                    if len(all_level_relation_class) > self.label_level:
                        relation_class = all_level_relation_class[self.label_level].lower()
                    else:
                        relation_class = None
                    possible_relation_classes = [item[self.label_level].lower() for item in possible_all_level_relation_classes if len(item) > self.label_level]
                    possible_relation_classes = [item for item in possible_relation_classes if item in self.label_list]
                    connectives = sample["conn"].split("##")[0].lower()

                    if (relation_class is None) or (relation_class not in self.label_list):
                        continue

                    # for encoder input_ids
                    tokens_1 = self.tokenizer.tokenize(arg1)
                    tokens_2 = self.tokenizer.tokenize(arg2)
                    tokens = ["<s>"] + tokens_1 + ["</s>", "</s>"] + tokens_2
                    if len(tokens) > self.max_seq_length - 1:
                        tokens = tokens[:self.max_seq_length - 1]
                    tokens = tokens + ["</s>"]
                    token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

                    arg_tokens_2 = self.tokenizer.tokenize("{} ".format(connectives.capitalize()) + arg2)
                    arg_tokens = ["<s>"] + tokens_1 + ["</s>", "</s>"] + arg_tokens_2
                    if len(arg_tokens) > self.max_seq_length - 1:
                        arg_tokens = arg_tokens[:self.max_seq_length - 1]
                    arg_tokens = arg_tokens + ["</s>"]
                    arg_token_ids = self.tokenizer.convert_tokens_to_ids(arg_tokens)

                    label_ids = self.label_list.index(relation_class)
                    possible_label_ids = np.zeros(len(self.label_list), dtype=np.int)
                    for label in possible_relation_classes:
                        possible_label_ids[self.label_list.index(label)] = 1

                    # padding
                    input_ids = np.ones(self.max_seq_length, dtype=np.int)
                    attention_mask = np.zeros(self.max_seq_length, dtype=np.int)
                    input_ids = input_ids * self.tokenizer.pad_token_id
                    input_ids[:len(token_ids)] = token_ids
                    attention_mask[:len(token_ids)] = 1

                    arg_input_ids = np.ones(self.max_seq_length, dtype=np.int)
                    arg_attention_mask = np.zeros(self.max_seq_length, dtype=np.int)
                    arg_input_ids = arg_input_ids * self.tokenizer.pad_token_id
                    arg_input_ids[:len(arg_token_ids)] = arg_token_ids
                    arg_attention_mask[:len(arg_token_ids)] = 1

                    all_input_ids.append(input_ids)
                    all_attention_mask.append(attention_mask)
                    all_arg_input_ids.append(arg_input_ids)
                    all_arg_attention_mask.append(arg_attention_mask)
                    all_label_ids.append(label_ids)
                    all_possible_label_ids.append(possible_label_ids)

        assert len(all_input_ids) == len(all_attention_mask), (len(all_input_ids), len(all_attention_mask))
        assert len(all_input_ids) == len(all_arg_input_ids), (len(all_input_ids), len(all_arg_input_ids))
        assert len(all_input_ids) == len(all_arg_attention_mask), (len(all_input_ids), len(all_arg_attention_mask))
        assert len(all_input_ids) == len(all_label_ids), (len(all_input_ids), len(all_label_ids))
        assert len(all_input_ids) == len(all_possible_label_ids), (len(all_input_ids), len(all_possible_label_ids))

        all_input_ids = np.array(all_input_ids)
        all_attention_mask = np.array(all_attention_mask)
        all_arg_input_ids = np.array(all_arg_input_ids)
        all_arg_attention_mask = np.array(all_arg_attention_mask)
        all_label_ids = np.array(all_label_ids)
        all_possible_label_ids = np.array(all_possible_label_ids)

        self.input_ids = all_input_ids
        self.attention_mask = all_attention_mask
        self.arg_input_ids = all_arg_input_ids
        self.arg_attention_mask = all_arg_attention_mask
        self.label_ids = all_label_ids
        self.possible_label_ids = all_possible_label_ids

        self.total_size = self.input_ids.shape[0]

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        return (
            torch.tensor(self.input_ids[index]),
            torch.tensor(self.attention_mask[index]),
            torch.tensor(self.arg_input_ids[index]),
            torch.tensor(self.arg_attention_mask[index]),
            torch.tensor(self.label_ids[index]),
            torch.tensor(self.possible_label_ids[index])
        )

