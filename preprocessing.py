# author = liuwei
# date = 2022-04-08

import json
import os
import re
import csv

def getConnLabel(text_array, is_altlex=False):
    array_size = len(text_array)
    boundary_id = -1
    for idx in range(1, array_size):
        text = text_array[idx]
        if "____" in text:
            boundary_id = idx
            break
    assert boundary_id > 0, (boundary_id)
    all_conns = []
    all_sences = []
    for idx in range(boundary_id-1, 1, -1):
        text = text_array[idx]
        if "####" in text:
            break
        if "Temporal" in text or "Contingency" in text or "Comparison" in text or "Expansion" in text:
            items = text.split(",")
            if is_altlex:
                for item in items:
                    all_sences.append(item.strip())
            else:
                all_conns.append(items[0].strip())
                for idx in range(1, len(items)):
                    all_sences.append(items[idx].strip())
        else:
            break

    conns = "##".join(all_conns)
    sences = "##".join(all_sences)

    return conns, sences

def getTextByName(text_array, str_name):
    array_size = len(text_array)
    start_pos = -1
    for idx in range(array_size):
        text = text_array[idx]
        if str_name in text:
            start_pos = idx
            break
    if start_pos == -1:
        print("No such attribution!!!")
        return None
    raw_text = []
    text_flag = False
    for idx in range(start_pos+1, array_size):
        text = text_array[idx]
        if "##########" in text:
            text_flag = False
            break
        if text_flag:
            text = text.replace("\n", "")
            raw_text.append(text)
        if "#### Text ####" in text:
            text_flag = True
        if "#### Features ####" in text:
            text_flag = False
            break

    return ", ".join(raw_text)

def pdtb2_sample_reader(text_array):
    relation_type = ""
    relation_class = ""
    conn = ""
    arg1 = ""
    arg2 = ""
    type_text =text_array[0]
    if "____Explicit____" in type_text:
        relation_type = "Explicit"
        conn, relation_class = getConnLabel(text_array)
    elif "____NoRel____" in type_text:
        relation_type = "NoRel"
    elif "____EntRel____" in type_text:
        relation_type = "EntRel"
    elif "____Implicit____" in type_text:
        relation_type = "Implicit"
        conn, relation_class = getConnLabel(text_array)
    elif "____AltLex____" in type_text:
        relation_type = "AltLex"
        conn, relation_class = getConnLabel(text_array, is_altlex=True)
    arg1 = getTextByName(text_array, "____Arg1____")
    arg2 = getTextByName(text_array, "____Arg2____")

    sample = {}
    sample["relation_type"] = relation_type
    sample["relation_class"] = relation_class
    sample["conn"] = conn
    sample["arg1"] = arg1
    sample["arg2"] = arg2

    return sample

def pdtb2_file_reader(input_file):
    all_samples = []
    with open(input_file, "r", encoding="ISO-8859-1") as f:
        lines = f.readlines()
        sample_boundaries = []
        for idx, line in enumerate(lines):
            line = line.strip()
            if "____Explicit____" in line or "____NoRel____" in line or "____EntRel____" in line \
                    or "____Implicit____" in line or "____AltLex____" in line:
                sample_boundaries.append(idx)

        sample_boundaries.append(idx) # add the last one
        boundary_size = len(sample_boundaries)
        for idx in range(boundary_size-1):
            sample_lines = lines[sample_boundaries[idx]:sample_boundaries[idx+1]]
            sample = pdtb2_sample_reader(sample_lines)
            all_samples.append(sample)
            # if sample["relation_type"] == "Implicit":
            #     all_samples.append(sample)

    return all_samples

def refine_raw_data_pdtb2(source_dir, data_list, output_dir, mode):
    """
    Args:
        source_dir:
        data_list: ["00", "01", ..., "02"]
        output_dir:
        mode: train, dev, test

    in paper "A pdtb-styled end-to-end discourse parser." 2-21 sections as training set,
    22 and 23 as develop and test respectively. in Ji and Eisenstein, 2015, 2-21 for training,
    0-1 as dev, 21-22 as test
    """
    data_dirs = [os.path.join(source_dir, data) for data in data_list]
    all_file_paths = []
    for data_dir in data_dirs:
        cur_files = os.listdir(data_dir)
        cur_files = [f for f in cur_files if ".pdtb" in f]
        cur_files = [os.path.join(data_dir, f) for f in cur_files]
        all_file_paths.extend(cur_files)

    out_file_name = "{}.json".format(mode)
    out_file_name = os.path.join(output_dir, out_file_name)
    all_samples = []
    all_file_paths = sorted(all_file_paths)
    for file_name in all_file_paths:
        print(file_name)
        cur_samples = pdtb2_file_reader(file_name)
        all_samples.extend(cur_samples)

    with open(out_file_name, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write("%s\n" % (json.dumps(sample, ensure_ascii=False)))

def pdtb3_file_reader(data_file, label_file):
    """
    Args:
        data_file: file path for raw text data
        label_file: label info for each data
    """
    all_samples = []

    with open(data_file, "r", encoding="latin1") as f: # utf-8
        text_data = f.read()

    with open(label_file, "r", encoding="latin1") as f: # utf-8
        lines = f.readlines()
        for line in lines:
            if line:
                items = line.split("|")

                relation_type = items[0].strip()
                conn1 = items[7].strip()
                conn1_sense1 = items[8].strip()
                conn2 = items[10].strip()
                conn2_sense1 = items[11].strip()

                arg1_idx = items[14].split(";")
                arg2_idx = items[20].split(";")
                arg1_str = []
                for pairs in arg1_idx:
                    arg1_i, arg1_j = pairs.split("..")
                    arg1 = text_data[int(arg1_i):int(arg1_j)+1]
                    arg1_str.append(re.sub("\n", " ", arg1))
                arg1 = ", ".join(arg1_str)

                arg2_str = []
                for pairs in arg2_idx:
                    if pairs == "":
                        continue
                    arg2_i, arg2_j = pairs.split("..")
                    arg2 = text_data[int(arg2_i):int(arg2_j)+1]
                    arg2_str.append(re.sub("\n", " ", arg2))
                arg2 = ", ".join(arg2_str)

                if int(arg1_idx[0].split("..")[0]) > int(arg2_idx[0].split("..")[0]):
                    tmp = arg1
                    arg1 = arg2
                    arg2 = tmp

                provenance = items[32].strip().lower()
                if "pdtb2" in provenance:
                    if "same" in provenance:
                        annotate_flag = "pdtb2.same"
                    elif "changed" in provenance:
                        annotate_flag = "pdtb2.changed"
                elif "pdtb3" in provenance:
                    annotate_flag = "pdtb3.new"

                sample = {}
                sample["relation_type"] = relation_type
                if conn1 and conn2:
                    sample["conn"] = conn1 + "##" + conn2
                elif conn1:
                    sample["conn"] = conn1
                elif conn2:
                    sample["conn"] = conn2
                else:
                    sample["conn"] = ""

                if conn1_sense1 and conn2_sense1:
                    sample["relation_class"] = conn1_sense1 + "##" + conn2_sense1
                elif conn1_sense1:
                    sample["relation_class"] = conn1_sense1
                elif conn2_sense1:
                    sample["relation_class"] = conn2_sense1
                else:
                    sample["relation_class"] = ""

                sample["arg1"] = arg1
                sample["arg2"] = arg2
                sample["annotate_flag"] = annotate_flag
                all_samples.append(sample)

    return all_samples

def refine_raw_data_pdtb3(source_dir, gold_dir, data_list, output_dir, mode):
    """
    Args:
        source_dir: raw data
        gold_dir: gold label
        data_list:
        output_dir:
        mode: train, dev, test

    We recall a existing model to convert the raw version of pdtb 3 to a tsv version.
    Please refer to paper: Implicit Discourse Relation Classification: We Need to Talk about Evaluation
    and also the github resposity of the model: https://github.com/najoungkim/pdtb3
    """
    all_file_paths = []
    for data in data_list:
        cur_data_dir = os.path.join(source_dir, data)
        cur_label_dir = os.path.join(gold_dir, data)
        cur_data_files = os.listdir(cur_data_dir)
        cur_label_files = os.listdir(cur_label_dir)
        cur_data_files = [f for f in cur_data_files if "wsj_" in f]
        cur_label_files = [f for f in cur_label_files if "wsj_" in f]
        for f in cur_label_files:
            if f in cur_data_files:
                all_file_paths.append((os.path.join(cur_data_dir, f), os.path.join(cur_label_dir, f)))

    out_file_name = "{}.json".format(mode)
    out_file_name = os.path.join(output_dir, out_file_name)
    all_samples = []
    all_file_paths = sorted(all_file_paths)
    for file_name in all_file_paths:
        print(file_name[0], file_name[1])
        cur_samples = pdtb3_file_reader(file_name[0], file_name[1])
        all_samples.extend(cur_samples)

    print(len(all_samples))
    with open(out_file_name, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write("%s\n" % (json.dumps(sample, ensure_ascii=False)))

def refine_raw_data_pcc(source_dir, output_dir):
    file_path = os.path.join(source_dir, "pcc_discourse_relations_all.tsv")
    csv_reader = csv.reader(open(file_path), delimiter="\t", quotechar='"')
    all_samples = []
    cur_line = 0
    for line in csv_reader:
        cur_line += 1
        if cur_line == 1:
            continue
        sample = {}
        sample["relation_type"] = line[3]
        sample["relation_class"] = line[2]
        sample["conn"] = line[4]
        sample["arg1"] = line[5]
        sample["arg2"] = line[6]
        all_samples.append(json.dumps(sample, ensure_ascii=False))

    random.shuffle(all_samples)
    total_num = len(all_samples)
    print(total_num)
    piece_num = int(total_num * 0.2)
    p1_samples = all_samples[:piece_num]
    p2_samples = all_samples[piece_num:piece_num * 2]
    p3_samples = all_samples[piece_num * 2:piece_num * 3]
    p4_samples = all_samples[piece_num * 3:piece_num * 4]
    p5_samples = all_samples[piece_num * 4:]
    all_pieces = [p1_samples, p2_samples, p3_samples, p4_samples, p5_samples]

    fold_group = [[0, 1, 2, 3, 4], [1, 2, 3, 4, 0], [2, 3, 4, 0, 1], [3, 4, 0, 1, 2], [4, 0, 1, 2, 3]]

    for fold_id in range(5):
        train_samples = []
        dev_samples = []
        test_samples = []

        train_samples.extend(all_pieces[fold_group[fold_id][0]])
        train_samples.extend(all_pieces[fold_group[fold_id][1]])
        train_samples.extend(all_pieces[fold_group[fold_id][2]])
        dev_samples.extend(all_pieces[fold_group[fold_id][3]])
        test_samples.extend(all_pieces[fold_group[fold_id][4]])
        print("Train size: %d, Dev size: %d, Test size: %d" % (len(train_samples), len(dev_samples), len(test_samples)))

        fold_dir = os.path.join(output_dir, "{}".format(fold_id+1))
        os.makedirs(fold_dir, exist_ok=True)
        train_file = os.path.join(fold_dir, "train.json")
        dev_file = os.path.join(fold_dir, "dev.json")
        test_file = os.path.join(fold_dir, "test.json")
        with open(train_file, "w", encoding="utf-8") as f:
            random.shuffle(train_samples)
            for text in train_samples:
                f.write("%s\n" % (text))

        with open(dev_file, "w", encoding="utf-8") as f:
            random.shuffle(dev_samples)
            for text in dev_samples:
                f.write("%s\n" % (text))

        with open(test_file, "w", encoding="utf-8") as f:
            random.shuffle(test_samples)
            for text in test_samples:
                f.write("%s\n" % (text))

        generate_label_file(fold_dir)


def write_labels_to_file(file_path, label_list):
    with open(file_path, "w", encoding="utf-8") as f:
        for label in label_list:
            f.write("%s\n"%(label))

def generate_label_file(data_dir):
    if "pdtb2" in data_dir:
        label1_file = os.path.join(data_dir, "labels_level_1.txt")
        label1_list = ["Comparison", "Contingency", "Expansion", "Temporal"]
        label2_file = os.path.join(data_dir, "labels_level_2.txt")
        label2_list = ['Asynchronous', 'Synchrony', 'Cause', 'Pragmatic Cause', 'Contrast', 'Concession', 'Conjunction', 'Instantiation', 'Restatement', 'Alternative', 'List']
        write_labels_to_file(label1_file, label1_list)
        write_labels_to_file(label2_file, label2_list)
    elif "pdtb3" in data_dir:
        label1_file = os.path.join(data_dir, "labels_level_1.txt")
        label1_list = ["Comparison", "Contingency", "Expansion", "Temporal"]
        label2_file = os.path.join(data_dir, "labels_level_2.txt")
        label2_list = ['Concession', 'Contrast', 'Cause', 'Cause+Belief', 'Condition', 'Purpose', 'Conjunction', 'Equivalence', 'Instantiation', 'Level-of-detail', 'Manner', 'Substitution', 'Asynchronous', 'Synchronous']
        write_labels_to_file(label1_file, label1_list)
        write_labels_to_file(label2_file, label2_list)
    elif "pcc" in data_dir:
        label2_file = os.path.join(data_dir, "labels_level_2.txt")
        label2_list = ["Cause", "Level-of-detail", "Conjunction", "Instantiation", "Contrast", "Equivalence", "Concession", "Asynchronous"]
        write_labels_to_file(label2_file, label2_list)

if __name__ == "__main__":
    #### PDTB2.0
    ## 1. Ji split
    source_dir = "data/dataset/pdtb2/raw"
    data_list = [
        "02", "03", "04", "05", "06", "07", "08", "09", "10", "11",
        "12", "13", "14", "15", "16", "17", "18", "19", "20"
    ]
    output_dir = "data/dataset/pdtb2/fine"
    os.makedirs(output_dir, exist_ok=True)
    mode = "train"
    refine_raw_data_pdtb2(source_dir=source_dir, data_list=data_list, output_dir=output_dir, mode=mode)

    data_list = ["00", "01"]
    mode = "dev"
    refine_raw_data_pdtb2(source_dir=source_dir, data_list=data_list, output_dir=output_dir, mode=mode)

    data_list = ["21", "22"]
    mode = "test"
    refine_raw_data_pdtb2(source_dir=source_dir, data_list=data_list, output_dir=output_dir, mode=mode)
    generate_label_file(output_dir)

    ## 2. Xval
    # X-validation
    sections = [
        '00', '01', '02', '03', '04', '05', '06', '07', '08',
        '09', '10', '11', '12', '13', '14', '15', '16', '17',
        '18', '19', '20', '21', '22', '23', '24'
    ]

    dev_sections = []
    test_sections = []
    train_sections = []

    for i in range(0, 25, 2):
        dev_sections.append([sections[i], sections[(i + 1) % 25]])
        test_sections.append([sections[(i + 23) % 25], sections[(i + 24) % 25]])
        train_sections.append([sections[(i + j) % 25] for j in range(2, 23)])
    print(dev_sections)
    print(test_sections)
    print(train_sections)
    for idx in range(12):
        source_dir = "data/dataset/pdtb2/raw"
        output_dir = "data/dataset/pdtb2/xval/fold_{}".format(idx + 1)
        os.makedirs(output_dir, exist_ok=True)
        mode = "train"
        refine_raw_data_pdtb2(source_dir, train_sections[idx], output_dir, mode)
        mode = "dev"
        refine_raw_data_pdtb2(source_dir, dev_sections[idx], output_dir, mode)
        mode = "test"
        refine_raw_data_pdtb2(source_dir, test_sections[idx], output_dir, mode)
        generate_label_file(output_dir)


    #### PDTB3.0
    ## 1. Ji split
    source_dir = "data/dataset/pdtb3/raw/data"
    gold_dir = "data/dataset/pdtb3/raw/gold"
    data_list = [
        "02", "03", "04", "05", "06", "07", "08", "09", "10", "11",
        "12", "13", "14", "15", "16", "17", "18", "19", "20"
    ]
    output_dir = "data/dataset/pdtb3/fine"
    os.makedirs(output_dir, exist_ok=True)
    mode = "train"
    # refine_raw_data_pdtb3(source_dir=source_dir, gold_dir=gold_dir, data_list=data_list, output_dir=output_dir, mode=mode)

    data_list = ["00", "01"]
    mode = "dev"
    # refine_raw_data_pdtb3(source_dir=source_dir, gold_dir=gold_dir, data_list=data_list, output_dir=output_dir, mode=mode)

    data_list = ["21", "22"]
    mode = "test"
    # refine_raw_data_pdtb3(source_dir=source_dir, gold_dir=gold_dir, data_list=data_list, output_dir=output_dir, mode=mode)
    # generate_label_file(output_dir)

    ## 2 Xval
    # X-validation
    sections = [
        '00', '01', '02', '03', '04', '05', '06', '07', '08',
        '09', '10', '11', '12', '13', '14', '15', '16', '17',
        '18', '19', '20', '21', '22', '23', '24'
    ]

    dev_sections = []
    test_sections = []
    train_sections = []

    for i in range(0, 25, 2):
        dev_sections.append([sections[i], sections[(i + 1) % 25]])
        test_sections.append([sections[(i + 23) % 25], sections[(i + 24) % 25]])
        train_sections.append([sections[(i + j) % 25] for j in range(2, 23)])

    for idx in range(12):
        source_dir = "data/dataset/pdtb3/raw/data"
        gold_dir = "data/dataset/pdtb3/raw/gold"
        output_dir = "data/dataset/pdtb3/xval/fold_{}".format(idx + 1)
        os.makedirs(output_dir, exist_ok=True)
        mode = "train"
        # refine_raw_data_pdtb3(source_dir=source_dir, gold_dir=gold_dir, data_list=train_sections[idx], output_dir=output_dir, mode=mode)
        mode = "dev"
        # refine_raw_data_pdtb3(source_dir=source_dir, gold_dir=gold_dir, data_list=dev_sections[idx], output_dir=output_dir, mode=mode)
        mode = "test"
        # refine_raw_data_pdtb3(source_dir=source_dir, gold_dir=gold_dir, data_list=test_sections[idx], output_dir=output_dir, mode=mode)
        # generate_label_file(output_dir)


    #### PCC
    ## Xval
    source_dir = "data/dataset/pcc/raw"
    output_dir = "data/dataset/pcc/xval"
    # refine_raw_data_pcc(source_dir, output_dir)
