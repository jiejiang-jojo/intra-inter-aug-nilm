#train
# python train.py --data_key 'm' --data_na "data-raw" --test_name 20

#python train.py --data_key 'd' --data_na "ccf-dishwasher-self-self" --test_name 16

#python train.py --data_key 'w' --data_na "ccf-washingmachine-self-self" --test_name 2

#python train.py --data_key 'k' --data_na "ccf-kettle-mixture" --test_name 2

#finetuned
# python train.py --data_key 'k' --data_na "UKDALE_bert_kettle" --test_name 25 --train_house 32

#python train.py --data_key 'd' --data_na "data-raw" --test_name 24 --train_house 20

python train.py --data_key 'm' --data_na "bmicrowave" --test_name 24  --train_house 20

#python train.py --data_key 'w' --data_na "bert-washignmachine-mixture-fine" --test_name 24 --train_house 20
