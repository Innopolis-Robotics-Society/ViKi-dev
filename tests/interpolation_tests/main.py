import sys
import json
from viki.optimization.interpolation.interpolation import Interpolator


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_interpolation.py <input.json>")
        sys.exit(1)

    input_file = sys.argv[1]

    with open(input_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    interpolator = Interpolator()
    result = interpolator.process(raw_data)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
