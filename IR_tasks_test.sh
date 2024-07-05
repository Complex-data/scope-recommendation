
MODEL_NAME=$1
CHECK=$(echo "$MODEL_NAME" | grep "Steam")
if [ "$CHECK" != "" ]; then
    ITEM_INDEX='title'
    DATASET='steam'
    PORT=24681
else
    ITEM_INDEX='title64_t'
    DATASET='sub_movie'
    PORT=24680
fi

python IR_test.py --data_path data/dataset/${DATASET}/ --item_index ${ITEM_INDEX} --model_name "$MODEL_NAME" --SFT_test_task SFTTestSeqRec --topk 10 --port ${PORT}
python IR_test.py --data_path data/dataset/${DATASET}/ --item_index ${ITEM_INDEX} --model_name "$MODEL_NAME" --SFT_test_task SFT+TestPersonalControlRec --topk 10 --port ${PORT}
python IR_test.py --data_path data/dataset/${DATASET}/ --item_index ${ITEM_INDEX} --model_name "$MODEL_NAME" --SFT_test_task SFT-TestPersonalControlRec --topk 10 --port ${PORT}
#python IR_test.py --data_path data/dataset/${DATASET}/ --item_index ${ITEM_INDEX} --model_name "$MODEL_NAME" --SFT_test_task SFTTestSeqRanking --topk 5 --candidate_num 10 --port ${PORT}
#python IR_test.py --data_path data/dataset/${DATASET}/ --item_index ${ITEM_INDEX} --model_name "$MODEL_NAME" --SFT_test_task SFTTestSeqRanking --topk 5 --candidate_num 10 --port ${PORT} --candidate_infer --batch_size 1
python IR_test.py --data_path data/dataset/${DATASET}/ --item_index ${ITEM_INDEX} --model_name "$MODEL_NAME" --SFT_test_task SFTTestPersonalCategoryRateLP_30 --topk 10 --port ${PORT}
python IR_test.py --data_path data/dataset/${DATASET}/ --item_index ${ITEM_INDEX} --model_name "$MODEL_NAME" --SFT_test_task SFTTestPersonalCategoryRateLP_50 --topk 10 --port ${PORT}
python IR_test.py --data_path data/dataset/${DATASET}/ --item_index ${ITEM_INDEX} --model_name "$MODEL_NAME" --SFT_test_task SFTTestPersonalCategoryRateLP_70 --topk 10 --port ${PORT}
