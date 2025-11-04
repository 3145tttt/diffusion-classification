from torchvision.transforms import Resize, Compose, Normalize, ToTensor, RandomHorizontalFlip, RandomGrayscale

from datasets import load_dataset
from transformers import DefaultDataCollator


id2label = {
    0: 'PixArt_Sigma',
    1: 'Kolors',
    2: 'SDXL_Turbo',
    3: 'Hyper_SD',
    4: 'SD_1.5',
    5: 'Realistic_Stock_Photo',
    6: 'Flash_PixArt',
    7: 'Flash_SD',
    8: 'SD_3',
    9: 'Lumina',
    10: 'Flash_SD3',
    11: 'SSD_1B',
    12: 'Kandinsky',
    13: 'Mobius',
    14: 'PixArt_Alpha',
    15: 'Flux_1',
    16: 'JuggernautXL',
    17: 'LCM_SDXL',
    18: 'LCM_SSD_1B',
    19: 'SDXL_Lightning',
    20: 'SD_2.1',
    21: 'Flash_SDXL',
    22: 'SD_Cascade',
    23: 'SDXL',
    24: 'IF'
    }

label2id = {
    'PixArt_Sigma': 0,
    'Kolors': 1,
    'SDXL_Turbo': 2,
    'Hyper_SD': 3,
    'SD_1.5': 4,
    'Realistic_Stock_Photo': 5,
    'Flash_PixArt': 6,
    'Flash_SD': 7,
    'SD_3': 8,
    'Lumina': 9,
    'Flash_SD3': 10,
    'SSD_1B': 11,
    'Kandinsky': 12,
    'Mobius': 13,
    'PixArt_Alpha': 14,
    'Flux_1': 15,
    'JuggernautXL': 16,
    'LCM_SDXL': 17,
    'LCM_SSD_1B': 18,
    'SDXL_Lightning': 19,
    'SD_2.1': 20,
    'Flash_SDXL': 21,
    'SD_Cascade': 22,
    'SDXL': 23,
    'IF': 24
}

def get_data(size, split, name='lesc-unifi/dragon'):
    # https://huggingface.co/datasets/lesc-unifi/dragon
    assert size in ["ExtraSmall", "Small", "Regular", "Large", "ExtraLarge"], f"size = {size}"
    assert split in ["train", "test"], f"split = {split}"

    dragon_dataset = load_dataset(name, size, split=split).select_columns(['model.txt', 'png'])
    return dragon_dataset

def create_base_transforms(base_size, image_processor, split):
    assert split in ["train", "validation"]

    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    if split == "train":
        return Compose([Resize(base_size), RandomHorizontalFlip(), RandomGrayscale(), ToTensor(), normalize])
    return Compose([Resize(base_size), ToTensor(), normalize])

def transforms(examples, _transforms):
    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["png"]]
    examples["labels"] = [label2id[txt] for txt in examples["model.txt"]]
    del examples["png"]
    del examples["model.txt"]
    return examples

def get_collator():
    data_collator = DefaultDataCollator()
    return data_collator