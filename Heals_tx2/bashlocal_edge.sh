rank=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9")
#ip=("33" "34" "35" "21" "22" "23" "24" "25" "29" "30")
for i in 0 1 2 3 4
do
sleep 2s
python edge.py --device_num 30 --model_type 'NIN' --dataset_type 'cifar100' --edge_id $i&
done
# for i in 5 6 7 8 9
# do
# sleep 2s
# python device.py --device_num 5 --node_num $i --use_gpu_id 0 &
# done
#pkill -f "device\.py --device_num*"

#python PS.py --device_num 30 --edge_number 5 --model_type 'NIN' --dataset_type 'cifar100' --alg_type 0
#python edge.py --device_num 3 --model_type 'AlexNet' --dataset_type 'cifar10' --edge_id 0
#python device.py --device_num 3 --edge_number 2 --node_num 0 --use_gpu_id 0 --model_type 'AlexNet' --dataset_type 'cifar10'