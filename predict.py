from PIL import Image
import torch
import os
from tqdm import tqdm
import pandas as pd
import yaml

import torch
from ml_collections import ConfigDict

from src.data import create_base_transforms
from src.model import get_model_pretraned, get_inference_model
from src.utils.random import set_global_seed

@torch.inference_mode()
def predict_img(model, img_tensor):
    logits = model(pixel_values=img_tensor)[0][0]
    pred_id = torch.argmax(logits).item()
    probs = logits.softmax(0)
    pred_type = model.config.id2label[pred_id]

    return  f"Pred diffusion = {pred_type}, with probability = {probs[pred_id].item():0.4f}"

@torch.inference_mode()
def predict_arr(model, config, input_path, output_path):
    with open(config) as stream:
        config = ConfigDict(yaml.safe_load(stream))
    set_global_seed(config.seed)

    _, image_processor = get_model_pretraned(config.model)
    test_transform = create_base_transforms(config.base_size, image_processor, 'test')

    img_paths = sorted(os.listdir(input_path))
    results = []
    for img_path in tqdm(img_paths):
        img = Image.open(os.path.join(input_path, img_path)).convert('RGB')
        img_tensor = test_transform(img)[None]
        res = predict_img(model, img_tensor)
        results.append(res)

    df = pd.DataFrame({'paths': img_paths, 'output': results})
    df.to_csv(output_path, index=False)


if __name__ == "__main__":

    repo_id = "3145tttt/diffusion-classification_base_resnet_50",
    model = get_inference_model(repo_id)
