rank=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24" "25" "26" "27" "28" "29")
#ip=("33" "34" "35" "21" "22" "23" "24" "25" "29" "30")
for i in 0 1 2 3 4
do
python device.py --device_num 30 --edge_number 5 --node_num $i --use_gpu_id 0 --model_type 'NIN' --dataset_type 'cifar100'&
done
for i in 5 6 7 8 9 10 11 12
do
python device.py --device_num 30 --edge_number 5 --node_num $i --use_gpu_id 4 --model_type 'NIN' --dataset_type 'cifar100'&
done
for i in 13 14 15 16 17 18 19 20
do
python device.py --device_num 30 --edge_number 5 --node_num $i --use_gpu_id 5 --model_type 'NIN' --dataset_type 'cifar100'&
done
for i in 21 22 23 24 25 26 27 28 29
do
python device.py --device_num 30 --edge_number 5 --node_num $i --use_gpu_id 6 --model_type 'NIN' --dataset_type 'cifar100'&
done
# for i in 5 6 7 8 9
# do
# sleep 2s
# python device.py --device_num 5 --node_num $i --use_gpu_id 0 &
# done
#pkill -f "edge\.py -device_num*"
#pkill -f "device\.py --device_num*"

#python PS.py --device_num 30 --edge_number 5 --model_type 'LeNet' --dataset_type 'emnist' --alg_type 3
#python edge.py --device_num 1 --model_type 'nin' --dataset_type 'image' --edge_id 0
#python device.py --device_num 1 --edge_number 1 --node_num 0 --use_gpu_id 0 --model_type 'nin' --dataset_type 'image'
#python PS.py --device_num 1 --edge_number 1 --model_type 'nin' --dataset_type 'image' --alg_type 0