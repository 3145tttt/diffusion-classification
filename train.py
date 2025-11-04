import comet_ml

import yaml
import click
import os
import torch
from ml_collections import ConfigDict

from src.utils.random import set_global_seed
from src.data import get_data, get_collator, create_base_transforms, transforms
from src.model import get_model_pretraned
from src.trainer import get_args, create_trainer, get_metric

@click.command()
@click.option(
    "--config",
    metavar="PATH",
    type=str,
    required=True,
    help="Path to config including model, training and dataset info.",
)
def main(
    config: str
):
    
    # Enable logging of model checkpoints
    use_comet = True
    if os.environ['COMET_API_KEY'] == 'YOUR API':
        print("Set comet API! Run withot logging")
        use_comet = False
    
    if use_comet:
        os.environ["COMET_LOG_ASSETS"] = "True"
        comet_ml.login(project_name="MLOPS")

    with open(config) as stream:
        config = ConfigDict(yaml.safe_load(stream))

    if not use_comet:
        config.train_conf.report_to = 'none'

    set_global_seed(config.seed)

    model, image_processor = get_model_pretraned(config.model)

    if torch.cuda.is_available():
        print("Use cuda!")
        device = "cuda:0"
        model.to(device)
    else:
        print("Use cpu!")
        device = "cpu"
    
    data_collator = get_collator()

    print("Prepare train data")
    train_ds = get_data(config.size, 'train')
    train_transform = create_base_transforms(config.base_size, image_processor, 'train')
    train_ds = train_ds.with_transform(lambda x: transforms(x, train_transform))

    print("Prepare test data")
    val_ds = get_data('ExtraSmall', 'test')
    test_transform = create_base_transforms(config.base_size, image_processor, 'test')
    val_ds = val_ds.with_transform(lambda x: transforms(x, test_transform))

    is_bf16_supported = (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    args = get_args(config.train_conf, is_bf16=is_bf16_supported)

    trainer = create_trainer(
        model=model, 
        training_args=args,
        data_collator=data_collator,
        train_dataset=train_ds, 
        eval_dataset=val_ds, 
        metric=get_metric(config.metric)
    )
    print("Start training")
    trainer.train()


if __name__ == "__main__":
    main()
