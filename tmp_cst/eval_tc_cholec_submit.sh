model="img_resnet101_bin"
for video in video01 video09 video12 video28 video52
do
    echo "Evaluating ${video}"
    mkdir -p /home/shunsuke/MICCAI2025/tmp_cst/plot/${model}/${video}
    python eval_tc_cholec.py --model ${model} --video_dir ${video} --save_plot_name ${video} > logs/${video}_${model}_tc.log 2>&1 &
done
wait