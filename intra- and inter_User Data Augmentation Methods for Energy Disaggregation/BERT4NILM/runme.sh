#Inference microwave
# for house in  17 18 19 20 ; do
# python train.py  --test_name=$house  --data_key 'm' --data_na "ccf-microwave-mixture"
# done

# for house in  17 18 19 20 ; do
# python train.py  --test_name=$house  --data_key 'm' --data_na "ccf-microwave-self-self"
# done

# for house in  17 18 19 20 ; do
# python train.py  --test_name=$house  --data_key 'm' --data_na "data"
# done



# for house in 16 18 20 ; do
# python train.py  --test_name=$house  --data_key 'd' --data_na "data-raw"
# done

# for house in  18 19 20; do
# python train.py  --test_name=$house  --data_key 'm' --data_na "data-raw"
# done

# for house in 18 19 20 ; do
# python train.py  --test_name=$house  --data_key 'w' --data_na "data-raw"
# done
# for house in  17 19 20; do
# python train.py  --test_name=$house  --data_key 'k' --data_na "ccf-kettle-self-self" 
# done
# for house in  17 ; do
# python train.py  --test_name=$house  --data_key 'k' --data_na "data-raw" 
# done


# for house in  26; do
# python train.py  --test_name=$house  --data_key 'd' --data_na "UKDALE_bert_dishwasher" --epo "1" --iter "2"
# done

for house in 24; do
python train.py  --test_name=$house  --data_key 'm' --data_na "bmicrowave" --epo "1" --iter "3"
done

# for house in 22; do
#python train.py    --data_key 'm' --data_na "bert-microwave-mixture-fine" --test_name 25 --train_house 20  --epo "1" --iter "9"
# done

# for house in 22 24 25 26; do
# python train.py  --test_name=$house  --data_key 'k' --data_na "UKDALE_kettle" 
# done


# for house in 22 23 26; do
# python train.py  --test_name=$house  --data_key 'd' --data_na "UKDALE_dishwasher" 
# done

# for house in 2; do
# python train.py  --test_name=$house  --data_key 'w' --data_na "ccf-washingmachine-paper" 
# done

# for house in 22 23; do
# python train.py  --test_name=$house  --data_key 'w' --data_na "UKDALE_washingmachine" 
# done