source ~/.bash_profile

# open the garbage collection feature in paddle 1.5
export FLAGS_eager_delete_tensor_gb=0.0

# set devices for training/predicting
# if you want to use CPU, please set CUDA_VISIBLE_DEVICES=""
export CUDA_VISIBLE_DEVICES=0,1

# training
python -u main.py \
    --do_train=True \
    --use_cuda=True \
    --do_eval_in_training=True \
    --eval_step=1000 \
    --save_model_path="./data/saved_models/" \
    --save_checkpoint="checkpoint" \
    --save_param="params"

# predicting and evaluating
python -u main.py \
    --do_predict=True \
    --do_eval=True \
    --init_from_params="./data/saved_models/params/step_final/" \
    --use_cuda=True

# saving the inference model
python -u main.py \
    --do_save_inference=True \
    --init_from_params="./data/saved_models/params/step_final/" \
    --use_cuda=True

