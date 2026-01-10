import yaml
from ml_collections import ConfigDict

from src.data import get_data, get_collator, create_base_transforms, transforms
from src.model import get_model_pretraned


CONFIG_TEST_RESNET = './configs/resnet_baseconf.yaml'

def test_forward(config=CONFIG_TEST_RESNET):
    """
    Test model forward
    """

    with open(config, 'r', encoding='utf-8') as stream:
        config = ConfigDict(yaml.safe_load(stream))

    print("Creating model ...")
    model, image_processor = get_model_pretraned(config.model)
    print("Creating model done")

    data_collator = get_collator()

    for data_type in ['train', 'test']:
        print(f"Creating data {data_type} ...")
        ds = get_data('ExtraSmall', data_type)
        ds = ds.with_transform(
            lambda x: transforms(
                x,
                create_base_transforms(config.base_size, image_processor, data_type)
            )
        )
        print(f"Creating data {data_type} done")

        batch = data_collator([ds[i] for i in range(2)])
        model(**batch)
        print(f"Forward for {data_type} done!")

if __name__ == "__main__":
    test_forward()
