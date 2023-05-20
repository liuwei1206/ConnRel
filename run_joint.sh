# For Ji split, change the dataset to pdtb2 or pdtb3
# 2
# <<"COMMENT"
python3 train_joint_conn_rel.py --do_train \
                                --dataset="other2" \
                                --label_file="labels_level_1.txt" \
                                --sample_k=100
# COMMENT
# 9

# 11
<<"COMMENT"
for seed in 106524 106464 106537 219539 430683
do
    python3 train_joint_conn_rel.py --do_train \
                                    --dataset="pdtb2" \
                                    --label_file="labels_level_1.txt" \
                                    --sample_k=100 \
                                    --seed=${seed}
done
COMMENT
# 22

sleep 10s

# 26
<<"COMMENT"
for seed in 106524 106464 106537 219539 430683
do
    python3 train_joint_conn_rel.py --do_train \
                                    --dataset="pdtb2" \
                                    --label_file="labels_level_2.txt" \
                                    --sample_k=100 \
                                    --seed=${seed}
done
COMMENT
# 37

sleep 10s

# For xval, change the dataset to pdtb2 or pdtb3
# 42
<<"COMMENT"
for idx in 1 2 3 4 5 6 7 8 9 10 11 12
do
    python3 train_joint_conn_rel.py --do_train \
                                    --dataset="pdtb2" \
                                    --fold_id=${idx} \
                                    --label_file="labels_level_1.txt" \
                                    --sample_k=100
done
COMMENT
# 53

sleep 10s

# 57
<<"COMMENT"
for idx in 1 2 3 4 5 6 7 8 9 10 11 12
do
    python3 train_joint_conn_rel.py --do_train \
                                    --dataset="pdtb2" \
                                    --fold_id=${idx} \
                                    --label_file="labels_level_2.txt" \
                                    --sample_k=100
done
COMMENT
# 68
