import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


if __name__ == "__main__":
    import cnnbin

    print("Testing import")
    print(cnnbin.__name__)

