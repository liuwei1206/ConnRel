# For Ji split, change the dataset to pdtb2 or pdtb3
# 2
# <<"COMMENT"
python3 train_roberta.py --do_train \
                         --dataset="test" \
                         --label_file="labels_level_1.txt" \
                         --use_conn \
                         --teacher_forcing
# COMMENT
# 10

# For xval, change the dataset to pdtb2 or pdtb3
# 13
<<"COMMENT"
for idx in 1 2 3 4 5 6 7 8 8 10 11 12
do
    python3 train_roberta.py --do_train \
                             --dataset="pdtb2" \
                             --fold_id=${idx} \
                             --label_file="labels_level_1.txt" \
                             --use_conn \
                             --teacher_forcing
done
COMMENT
# 25
