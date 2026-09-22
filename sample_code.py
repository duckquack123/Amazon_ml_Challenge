"""Create a valid baseline submission for the product pricing challenge."""

import argparse
from pathlib import Path

import pandas as pd


def create_baseline_submission(train_path: Path, test_path: Path, output_path: Path) -> None:
    """Predict the training-set median price for every test sample."""
    train = pd.read_csv(train_path, usecols=["price"])
    test = pd.read_csv(test_path, usecols=["sample_id"])

    if train["price"].dropna().empty:
        raise ValueError("The training file does not contain any usable prices.")

    baseline_price = float(train["price"].median())
    output = test.assign(price=max(baseline_price, 0.01))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Saved {len(output):,} predictions to {output_path}")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=root / "dataset" / "train.csv")
    parser.add_argument("--test", type=Path, default=root / "dataset" / "test.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "submission" / "baseline_submission.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_baseline_submission(args.train, args.test, args.output)
