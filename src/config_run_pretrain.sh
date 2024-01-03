need_run_config_arr=()
need_run_config_arr+=('1e-4-o_madr-mul-softmax-a01-fix')
for config_name in ${need_run_config_arr[@]}
do
    echo $config_name
    output_path='/path/output/'$config_name
    config_path='/path/config_pretrain/'$config_name'.yaml'
    CUDA_VISIBLE_DEVICES=0,1 python -m tevatron.driver.pretrain \
        --do_train \
        --output_dir ${output_path} \
        --tensorboard_output_dir ${output_path}/tboard \
        --model_name_or_path '' \
        --overwrite_output_dir \
        --fp16 \
        --config_path $config_path \
        > $config_name.log 2>&1
done
sleep 2m
nohup sh pretrain_finetune1_c.sh > 1e-4-o_madr-mul-softmax-a01-fix#f1=sh#f2=con-no#f3=madr_e1416.log 2>&1 &
nohup sh pretrain_finetune1.sh > 1e-4-o_madr-mul-softmax-a01-fix#f1=sh#f2=con-no#f3=madr_e1820.log 2>&1 &