import yaml
import torch
from ml_collections import ConfigDict

from transformers import AutoModelForImageClassification

from src.data import get_data, create_base_transforms
from src.model import get_model_pretraned
from src.utils.random import set_global_seed

def get_inference_model(repo_id):
    return AutoModelForImageClassification.from_pretrained(repo_id)

@torch.inference_mode()
def inference_model(
    test_object,
    config_path="./configs/resnet_baseconf.yaml",
    repo_id="3145tttt/diffusion-classification_base_resnet_50",
):
    """
    Inference model on test image
    """
    with open(config_path) as stream:
        config = ConfigDict(yaml.safe_load(stream))

    set_global_seed(config.seed)
    model = get_inference_model(repo_id)
    _, image_processor = get_model_pretraned(config.model)

    test_transform = create_base_transforms(config.base_size, image_processor, 'test')

    logits = model(pixel_values=test_transform(test_object['png'])[None])[0][0]
    pred_id = torch.argmax(logits).item()
    probs = logits.softmax(0)
    pred_type = model.config.id2label[pred_id]
    true_type = test_object['model.txt']

    print(f"Pred diffusion = {pred_type}, with probability = {probs[pred_id].item():0.4f}")
    print(f"True diffusion = {true_type}")


if __name__ == "__main__":
    test_object = get_data('ExtraSmall', 'test')[617]
    inference_model(test_object)

# Output:
# Pred diffusion = SD_1.5, with probability = 0.9487
# True diffusion = SD_1.5
