source ~/.bash_profile

# open garbage collection to save memory
export FLAGS_eager_delete_tensor_gb=0.0

# setting visible devices for training
export CUDA_VISIBLE_DEVICES=0,1,2,3

# training
python -u main.py \
    --do_train=False \
    --epoch=2 \
    --learning_rate=3e-5 \
    --weight_decay=0.001 \
    --use_cuda=True \
    --init_from_pretrain_model="./data/pretrain_models/bert_large_cased/params/" \
    --save_model_path="./data/saved_models/" \
    --save_checkpoint="checkpoints" \
    --save_param="params" \
    --training_file="./data/input/train-v1.1.json" \
    --output_prediction_file="./data/output/predictions.json" \
    --output_nbest_file="./data/output/nbest_predictions.json" \
    --bert_config_path="./data/pretrain_models/bert_large_cased/bert_config.json" \
    --vocab_path="./data/pretrain_models/bert_large_cased/vocab.txt" \
    --max_seq_len=384 

# setting visible devices for predicting
export CUDA_VISIBLE_DEVICES=0

# predicting
python -u main.py \
    --do_predict=True \
    --use_cuda=True \
    --init_from_params="./data/saved_models/params/step_final/" \
    --bert_config_path="./data/pretrain_models/bert_large_cased/bert_config.json" \
    --vocab_path="./data/pretrain_models/bert_large_cased/vocab.txt" \
    --output_prediction_file="./data/output/predictions.json" \
    --output_nbest_file="./data/output/nbest_predictions.json" \
    --prediciton_dir="./data/output/" \
    --predict_file="./data/input/dev-v1.1.json" \
    --max_seq_len=384

# setting visible devices for evaluating
export CUDA_VISIBLE_DEVICES=""

# evaluating
python -u main.py \
    --do_eval=True \
    --evaluation_file="./data/input/dev-v1.1.json" \
    --output_prediction_file="./data/output/predictions.json" 

# setting visible devices for predicting
export CUDA_VISIBLE_DEVICES=0

# saving the inference model
python -u main.py \
    --do_save_inference=True \
    --use_cuda=True \
    --init_from_pretrain_model="./data/pretrain_models/bert_large_cased/params/" \
    --bert_config_path="./data/pretrain_models/bert_large_cased/bert_config.json" \
    --inference_model_dir="./data/inference_model/" \
    --max_seq_len=384
