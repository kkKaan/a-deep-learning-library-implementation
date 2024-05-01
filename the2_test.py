import pandas as pd
import io
import numpy as np
import torch
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

    g1 = rastgele_dogal((2, 2))

    # generate 2x2 numpy array and take backward pass
    ones = [[1, 1], [1, 1], [1, 1]]
    array = np.array([[1, 2], [3, 4], [5, 6]])
    gArray = gergen([[1, 2], [3, 4], [5, 6]])
    tArray = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32, requires_grad=True)
    # a = np.gradient(array.mean())
    # print(a)

    # print(array.mean(axis = 1))
    # ort = Ortalama()
    # res = ort.ileri(gArray, eksen = 1)
    # print(res)
    # tMean = torch.mean(tArray, dim = 1)
    # print(tMean)

    # # Take geri pass
    # print(ort.geri(gergen(ones)))

    # # Loss calculation (using a simple target for demonstration)
    # target = torch.tensor([1, 1, 1], dtype = torch.float32)
    # loss = (tMean - target).pow(2).sum()

    # # Backward pass
    # loss.backward()

    # # Print gradients
    # print(tArray.grad)

    # test ileri of a operation
    # gArray = gArray * 2
    # print(gArray.operation.b)
    # print(gArray.operation.geri(gergen(ones))[0])
    # print(gArray.operation.ileri(gArray, gergen(ones) * 4))

    # test geri of a operation
    # print(gArray.operation.b)
    # print(gArray.operation.geri(gergen(ones))[1])

    #### test torch outer product
    # print("torch outer product")
    # t1 = torch.tensor([1, 2, 3], dtype=torch.float32, requires_grad=True)
    # t2 = torch.tensor([4, 5, 6], dtype=torch.float32, requires_grad=True)
    # outer = torch.outer(t1, t2)
    # grad_tensor = torch.ones_like(outer)
    # print("outer product of torch.outer: \n", outer)
    # # print(t1.grad) # -> will return None because grad of outer is not calculated yet
    # outer.backward(grad_tensor)
    # print("gradient of the first tensor: \n", t1.grad)
    # print("gradient of the second tensor: \n", t2.grad)

    # print("#############################################")

    # #### test geri of Dış Çarpım
    # print("geri of Dış Çarpım")
    # ones3_3 = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    # g1 = gergen([1, 2, 3])
    # g2 = gergen([4, 5, 6])
    # dis = DisCarpim()
    # res = dis.ileri(g1, g2)
    # print("outer product of geri: \n", res)
    # back = dis.geri(gergen(ones3_3))
    # print("gradient of the first tensor: \n", back[0])
    # print("gradient of the second tensor: \n", back[1])


if __name__ == "__main__":
    cekirdek(2)
    # data_file = "./path_to_data"
    # main(data_file)
    main()
