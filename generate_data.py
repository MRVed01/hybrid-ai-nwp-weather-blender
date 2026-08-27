from src.config import DATA_FILE
from src.data_generation import generate_synthetic_weather, save_dataset

if __name__ == "__main__":
    df = generate_synthetic_weather()
    save_dataset(df, DATA_FILE)
    print(f"Generated {len(df):,} rows -> {DATA_FILE}")
