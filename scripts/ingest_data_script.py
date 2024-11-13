import argparse

from my_package import ingest_data


def main():
    parser = argparse.ArgumentParser(description="Ingest Housing Data")
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Directory to save the dataset",
    )
    args = parser.parse_args()

    # Fetch and load the housing data
    ingest_data.fetch_housing_data(housing_path=args.output_path)
    housing = ingest_data.load_housing_data(housing_path=args.output_path)
    print("Data fetched and saved at:", args.output_path)
    print(housing.head())


if __name__ == "__main__":
    main()
