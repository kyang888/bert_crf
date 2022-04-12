PROJ_PATH=/root/autodl-tmp/bert_crf
cd ${PROJ_PATH} # enter the project path
echo "Enter project path: ${PROJ_PATH}"
export PYTHONPATH=$PROJ_PATH
MODEL_NAMES=bert_crf
SAVE_CKPT_INTERVAL=50
DATA_NAMES=(clue)
EXPS=(wwm_ext)
# roberta_wwm_ext wwm_ext wwm roberta_zh
#                            --use_lstm \
#                            --lstm_bidirectional \
#                            --lstm_layers_num=1 \
for DATA_NAME in  ${DATA_NAMES[@]}
do
  for EXP in ${EXPS[@]}
  do
    DIR_LOG="./logs/${DATA_NAME}/${EXP}"
    mkdir -p ${DIR_LOG}
    chmod -R 777 ${DIR_LOG}
    python3 train.py --dir_init_checkpoint="./init_ckpt/${EXP}/pytorch_model.bin" \
                            --ner_file="./resources/ner2id.json" \
                            --num_labels=31 \
                            --seed=123 \
                            --file_vocab="./resources/vocab.txt" \
                            --learning_rate=1e-5 \
                            --warmup_epoch=2.0 \
                            --num_train_step=1600 \
                            --batch_size_train=64 \
                            --batch_size_dev=100 \
                            --dir_training_data="./data/${DATA_NAME}" \
                            --dir_checkpoint="./checkpoint/${DATA_NAME}/${EXP}" \
                            --dir_best_model="./checkpoint/${DATA_NAME}/${EXP}/best_model" \
                            --dir_summary="./summary/${DATA_NAME}/${EXP}" \
                            --dir_log=${DIR_LOG} \
                            --gradient_accumulation_steps=1 \
                            --interval_print_info=20 \
                            --save_checkpoint_interval=${SAVE_CKPT_INTERVAL} \
                            --max_num_best_model=1 \
                            --max_num_checkpoint=1
  done
done