import pandas as pd
import io
from sklearn.preprocessing import LabelBinarizer

from gergen import *
from the2 import *

def test(mlp, inputs, targets):
    """
    TODO: Implement the testing pipeline
    """
    loss = None

    print("Test Loss: {}".format(loss))
    return loss

def data_preprocessing(data_file):
    """
    TODO:    DATA PREPROCESSING
    """
    # Load the data

    # Get the first column as labels (You can use one-hot encoding if needed (You can use sklearn or pandas for this))

    # Get the remaining columns as data

    # Return the data and labels
    return None, None
    

def main():
    # Some basic tests for gergen class
    # g1 = rastgele_dogal((2,2))
    # g2 = gergen([1,2,3])
    # print(g1.us(2))
    # print(g2.us(2))

    g1 = rastgele_dogal((2,2))

if __name__ == "__main__":
    cekirdek(2)
    # data_file = "./path_to_data"
    # main(data_file)
    main()