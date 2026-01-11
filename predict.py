from PIL import Image
import torch
import os
from tqdm import tqdm
import pandas as pd
import yaml
import click

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

    return pred_type, probs[pred_id].item()

@torch.inference_mode()
def predict_arr(model, config, input_path, output_path):
    with open(config) as stream:
        config = ConfigDict(yaml.safe_load(stream))
    set_global_seed(config.seed)

    _, image_processor = get_model_pretraned(config.model)
    test_transform = create_base_transforms(config.base_size, image_processor, 'test')

    img_paths = sorted(os.listdir(input_path))
    results_res = []
    results_prob = []
    for img_path in tqdm(img_paths):
        img = Image.open(os.path.join(input_path, img_path)).convert('RGB')
        img_tensor = test_transform(img)[None]
        pred, prob = predict_img(model, img_tensor)
        results_res.append(pred)
        results_prob.append(prob)

    df = pd.DataFrame({'paths': img_paths, 'output': results_res, 'prob': results_prob})
    df.to_csv(output_path, index=False)


@torch.inference_mode()
def inference_model(
    img,
    target,
    model,
    config
):
    """
    Inference model on test image
    """
    with open(config) as stream:
        config = ConfigDict(yaml.safe_load(stream))

    set_global_seed(config.seed)
    _, image_processor = get_model_pretraned(config.model)

    test_transform = create_base_transforms(config.base_size, image_processor, 'test')
    img_tensor = test_transform(img)[None]
    pred, prob = predict_img(model, img_tensor)
    
    print(f"Pred diffusion = {pred}, with probability = {prob:0.4f}")
    print(f"True diffusion = {target}")



@click.command()
@click.option(
    "--config",
    metavar="PATH",
    type=str,
    required=True,
    help="Path to config including model, training and dataset info.",
    default='./configs/resnet_baseconf.yaml'
)
@click.option(
    "--input_path",
    metavar="PATH",
    type=str,
    required=True,
    help="Path to config including model, training and dataset info."
)
@click.option(
    "--output_path",
    metavar="PATH",
    type=str,
    required=True,
    help="Path to config including model, training and dataset info."
)
def main(
    config: str,
    input_path: str,
    output_path: str,
):
    print(f"Используется config {config.split('/')[-1]}\n")
    with open(config) as stream:
        repo_id = ConfigDict(yaml.safe_load(stream)).train_conf.run_name
    try:
        last_checkpoint = sorted(os.listdir(repo_id))[-1]
        repo_id = f"{repo_id}/{last_checkpoint}"
        print(f"Выбран чекпоинт {repo_id}")
        model = get_inference_model(repo_id)
    except Exception:
        print(f"Не найдена обученная модель в папке {repo_id}")
        print("Будет использована предобученная модель с Hugging Face")
        print("Путь до модели: 3145tttt/diffusion-classification_base_resnet_50")
        print("Если вы хотите поменять путь до модели, выберите нужный config через изменение --config")
        repo_id = "3145tttt/diffusion-classification_base_resnet_50"
        model = get_inference_model(repo_id)

    predict_arr(model, config, input_path, output_path)

if __name__ == "__main__":
    main()