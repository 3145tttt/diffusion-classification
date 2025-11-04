# conda deactivate && conda activate diffclass
export COMET_API_KEY='YOUR API'
CUDA_VISIBLE_DEVICES=0 python train.py --config ./configs/resnet_baseconf.yaml