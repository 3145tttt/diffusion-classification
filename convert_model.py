import torch

from src.model import get_inference_model

repo_id = "3145tttt/diffusion-classification_base_resnet_50"
model = get_inference_model(repo_id)
torch.save(model, "model_full.pth")