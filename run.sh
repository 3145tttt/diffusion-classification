# cd NLP/ARM-LM/
# conda activate nlp
export COMET_API_KEY='YOUR API'
CUDA_VISIBLE_DEVICE=0 python train.py --config ./configs/resnet_baseconf.yaml