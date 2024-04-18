for sub in {1..16}
do
    subject=$(printf "[%d]" $sub)
    CUDA_VISIBLE_DEVICES=0 python -Wignore ./train/train_direction_with_val.py --subject $subject --strategy 2 --dataset KUL --lr 0.001 --model DenseNet_3D --win 3  --epoch 50 --batch 16  --stride 3 --topo True # --prototype 1
done