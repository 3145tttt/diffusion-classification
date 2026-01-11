import yaml
import torch
from ml_collections import ConfigDict
from datasets import load_from_disk

from src.data import create_base_transforms
from src.model import get_model_pretraned, get_inference_model
from src.utils.random import set_global_seed


@torch.inference_mode()
def inference_model(
    test_object,
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

    logits = model(pixel_values=test_transform(test_object['png'])[None])[0][0]
    pred_id = torch.argmax(logits).item()
    probs = logits.softmax(0)
    pred_type = model.config.id2label[pred_id]
    true_type = test_object['model.txt']

    print(f"Pred diffusion = {pred_type}, with probability = {probs[pred_id].item():0.4f}")
    print(f"True diffusion = {true_type}")


if __name__ == "__main__":
    test_object = load_from_disk('./datasets/ExtraSmall_test')[617]
    repo_id = "3145tttt/diffusion-classification_base_resnet_50",
    model = get_inference_model(repo_id)
    inference_model(test_object, model)

# Output:
# Pred diffusion = SD_1.5, with probability = 0.9487
# True diffusion = SD_1.5
