from transformers import AutoModelForImageClassification

PATH = "./base_resnet_50_pretrained/checkpoint-2000"
model = AutoModelForImageClassification.from_pretrained(PATH)
model.push_to_hub(
    repo_id='3145tttt/diffusion-classification_base_resnet_50',
    commit_message="Upload trained model weights",
    safe_serialization=True
)
