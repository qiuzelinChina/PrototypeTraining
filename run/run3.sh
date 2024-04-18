for k in 1 2 3 4
do
    for win in 3 5
    do
        for strategy in 1 3
        do
            
            for sub in {1..16}
            do
                subject=$(printf "[%d]" $sub)
                CUDA_VISIBLE_DEVICES=3 python ./train/train_direction_with_val.py --subject $subject --epoch 50 --strategy $strategy --dataset KUL --lr 0.001 --model CNN_base_fft  --batch 16 --win $win --stride 1 
            done

            for sub in {1..18}
            do
                subject=$(printf "[%d]" $sub)
                CUDA_VISIBLE_DEVICES=3 python ./train/train_direction_with_val.py --subject $subject --epoch 50 --strategy $strategy --dataset DTU --lr 0.001 --model CNN_base_fft  --batch 16 --win $win  --stride 1 
            done

            for sub in $(seq 21 44 | grep -v 24)
            do
                subject=$(printf "[%d]" $sub)
                CUDA_VISIBLE_DEVICES=3 python ./train/train_direction_with_val.py --subject $subject --epoch 50 --strategy $strategy --dataset DS --lr 0.001 --model CNN_base_fft --batch 16 --win $win  --stride 1
            done
        


            for sub in {1..16}
            do
                subject=$(printf "[%d]" $sub)
                CUDA_VISIBLE_DEVICES=3 python ./train/train_direction_with_val.py --subject $subject --epoch 50 --strategy 2 --dataset KUL --lr 0.001 --model CNN_base_fft  --batch 16 --win $win --stride $win 
            done

            for sub in {1..18}
            do
                subject=$(printf "[%d]" $sub)
                CUDA_VISIBLE_DEVICES=3 python ./train/train_direction_with_val.py --subject $subject --epoch 50 --strategy 2 --dataset DTU --lr 0.001 --model CNN_base_fft  --batch 16 --win $win  --stride $win
            done

            for sub in $(seq 21 44 | grep -v 24)
            do
                subject=$(printf "[%d]" $sub)
                CUDA_VISIBLE_DEVICES=3 python ./train/train_direction_with_val.py --subject $subject --epoch 50 --strategy 2 --dataset DS --lr 0.001 --model CNN_base_fft --batch 16 --win $win  --stride $win
            done
        done
    done
done