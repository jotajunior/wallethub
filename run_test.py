from build_model import test_model_from_file
from parse_data import parse_testing
import sys

if __name__ == '__main__':
    filename = sys.argv[1]
    x, y = parse_testing(filename)
    test_model_from_file(x, y, 1)
