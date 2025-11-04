import torch

from transformers import AutoModelForImageClassification, AutoImageProcessor
from src.data import id2label, label2id

def get_model_pretraned(checkpoint):
    assert checkpoint in ["microsoft/resnet-50", "google/vit-base-patch16-224"], f"{checkpoint}"
    assert len(id2label) == len(label2id), f"{id2label}, {label2id}"

    if checkpoint == "google/vit-base-patch16-224":
        model = AutoModelForImageClassification.from_pretrained(
            checkpoint,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
        )
    else:
        model = AutoModelForImageClassification.from_pretrained(checkpoint)

        if checkpoint == "microsoft/resnet-50":
            model.classifier[1] = torch.nn.Linear(2048, len(id2label))
            model.config.id2label = id2label

        model.config.id2label = id2label
        model.config.label2id = label2id

    model_size = sum(t.numel() for t in model.parameters())
    print(f"Model size: {model_size/1000**2:.1f}M parameters")

    return model, AutoImageProcessor.from_pretrained(checkpoint, use_fast=True)
