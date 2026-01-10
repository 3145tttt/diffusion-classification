from src.data import get_data


if __name__ == "__main__":
    SIZE = "ExtraSmall"
    for data_type in ['train', 'test']:
        print(f"Saving data {data_type} ...")
        ds = get_data(SIZE, data_type)
        ds.save_to_disk(f'./local_{SIZE}_{data_type}')