model="resnet101_bin"
for video in video_28 video_11_2
do
    echo "Evaluating ${video}"
    mkdir -p /home/shunsuke/MICCAI2025/tmp_cst/plot/${model}/${video}
    cp /home/shunsuke/MICCAI2025/tmp_cst/plot/tf_efficientnet_b7/${video}/gt.npy /home/shunsuke/MICCAI2025/tmp_cst/plot/${model}/${video}/gt.npy
    python eval_tc.py --model ${model} --video_dir /mnt/ssd1/EndoVis2022/train/${video} --save_plot_name ${video} > logs/${video}_${model}_1.log 2>&1 &
done


