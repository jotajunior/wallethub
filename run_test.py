from build_model import test_model_from_file
from parse_data import parse_testing
import sys

if __name__ == '__main__':
    filename = sys.argv[1]
    
    if len(sys.argv) < 3:
        n = 1
    else:
        n = int(sys.argv[2])

    x, y = parse_testing(filename)
    test_model_from_file(x, y, n)
