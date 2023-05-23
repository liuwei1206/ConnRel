### 1. For Ji split, change the dataset to pdtb2 or pdtb3
# 2
<<"COMMENT"
python3 train_pipeline.py --do_train \
                          --dataset="test" \
                          --label_file="labels_level_1.txt"
COMMENT
# 8

# 10
# <<"COMMENT"
for seed in 106524 106464 106537 219539 430683
do
    python3 train_pipeline.py --do_train \
                              --dataset="pdtb2" \
                              --label_file="labels_level_1.txt" \
                              --seed=${seed} \
                              --use_conn
done
# COMMNET
# 21

sleep 10s

# 25
# <<"COMMENT"
for seed in 106524 106464 106537 219539 430683
do
    python3 train_pipeline.py --do_train \
                              --dataset="pdtb2" \
                              --label_file="labels_level_2.txt" \
                              --seed=${seed} \
                              --use_conn
done
# COMMENT
# 36

sleep 10s

### 2. For xval, change the dataset to pdtb2 or pdtb3
# 41
# <<"COMMENT"
for idx in 1 2 3 4 5 6 7 8 10 11 12
do
    python3 train_pipeline.py --do_train \
                              --dataset="pdtb2" \
                              --fold_id=${idx} \
                              --label_file="labels_level_1.txt" \
                              --use_conn
done
# COMMENT
# 52

sleep 10s

# 56
# <<"COMMENT"
for idx in 1 2 3 4 5 6 7 8 10 11 12
do
    python3 train_pipeline.py --do_train \
                              --dataset="pdtb2" \
                              --fold_id=${idx} \
                              --label_file="labels_level_2.txt" \
                              --use_conn
done
# COMMENT
# 67
