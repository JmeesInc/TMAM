model="maxvit_tiny_tf_512"
for video in video_28 video_11_2
do
    echo "Evaluating ${video}"
    mkdir -p plot/${model}/${video}
    python eval_tc.py --model ${model} --video_dir EndoVis2022/train/${video} --save_plot_name ${video} > logs/${video}_${model}_1.log 2>&1 &
done


