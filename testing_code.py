import yaml
from ml_collections import ConfigDict
import traceback

from src.data import get_data, get_collator, create_base_transforms, transforms
from src.model import get_model_pretraned, get_inference_model
from predict import predict_arr, inference_model


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

def test_inference(config_path=CONFIG_TEST_RESNET):
    with open(config_path, 'r', encoding='utf-8') as stream:
        config = ConfigDict(yaml.safe_load(stream))

    print("Load data...")
    test_object = get_data('ExtraSmall', 'test')[617]
    print("Load data done")

    print("Creating model ...")
    model, _ = get_model_pretraned(config.model)
    print("Creating model done")

    inference_model(test_object['png'], test_object['model.txt'], model, config_path)

def test_predict(config_path=CONFIG_TEST_RESNET):
    with open(config_path, 'r', encoding='utf-8') as stream:
        config = ConfigDict(yaml.safe_load(stream))

    print("Creating model ...")
    model, _ = get_model_pretraned(config.model)
    print("Creating model done")

    predict_arr(model, config_path, './testing_imgs', 'testing_results.csv')

def test_pretrained_model(config_path=CONFIG_TEST_RESNET):
    test_object = get_data('ExtraSmall', 'test')[617]
    repo_id = "3145tttt/diffusion-classification_base_resnet_50"
    model = get_inference_model(repo_id)
    inference_model(test_object['png'], test_object['model.txt'], model, config_path)

if __name__ == "__main__":

    try:
        print("=" * 10)
        print("Test Forward")
        test_forward()
    except Exception:
        print("Test Forward error:")
        traceback.print_exc()
    else:
        print("Test Forward pass")
    print("\n\n")

    try:
        print("=" * 10)
        print("Test inference")
        test_inference()
    except Exception:
        print("Test inference error:")
        traceback.print_exc()
    else:
        print("Test inference pass")
    print("\n\n")
    
    try:
        print("=" * 10)
        print("Test predict")
        test_predict()
    except Exception:
        print("Test predict error:")
        traceback.print_exc()
    else:
        print("Test predict pass")
    print("\n\n")


    try:
        print("=" * 10)
        print("Test pretrained model")
        test_pretrained_model()
    except Exception:
        print("Test pretrained model error:")
        traceback.print_exc()
    else:
        print("Test pretrained model pass")
    print("\n\n")


