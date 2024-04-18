
for sub in {1...16}
do
    subject=$(printf "[%d]" $sub)
    CUDA_VISIBLE_DEVICES=0 python ./train/train_direction_with_val.py --subject $subject --epoch 50 --strategy 1 --dataset Das2016 --lr 0.001 --model DenseNet_3D --topo True --batch 16 --win 1 --stride 1 --band [14,31]
done

for sub in {1...18}
do
    subject=$(printf "[%d]" $sub)
    CUDA_VISIBLE_DEVICES=0 python ./train/train_direction_with_val.py --subject $subject --epoch 50 --strategy 1 --dataset Fug2018 --lr 0.001 --model DenseNet_3D --topo True --batch 16 --win 1 --stride 1 --band [14,31]
done

for sub in 21 22 23 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 43 44
do
    subject=$(printf "[%d]" $sub)
    CUDA_VISIBLE_DEVICES=0 python ./train/train_direction_with_val.py --subject $subject --epoch 50 --strategy 1 --dataset Fug2020 --lr 0.001 --model DenseNet_3D --topo True --batch 16 --win 1 --stride 1 --band [14,31]
done