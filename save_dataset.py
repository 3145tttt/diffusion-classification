import yaml
import click
from ml_collections import ConfigDict

from src.data import get_data

@click.command()
@click.option(
    "--config",
    metavar="PATH",
    type=str,
    required=True,
    help="Path to config including model, training and dataset info.",
)
def main(
    config: str = "./resnet_baseconf.yaml"
):
    with open(config) as stream:
        config = ConfigDict(yaml.safe_load(stream))

    for data_type in ['train', 'test']:
        print(f"Saving data {data_type} ...")
        ds = get_data(config.size, data_type)
        ds.save_to_disk(f'./datasets/{config.size}_{data_type}')

if __name__ == "__main__":
    main()
