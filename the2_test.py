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

    #########################################################################

    # print("-------Ic Carpim-------")

    # print("---ileri---")

    # # gergen ic carpim
    # print("gergen ic carpim")
    # g1 = gergen([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # g2 = gergen([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # ic = IcCarpim()
    # res = ic.ileri(g1, g2)
    # print("result of gergen ic carpim: \n", res)

    # print("#############################################")

    # # torch ic carpim
    # print("torch ic carpim")
    # t1 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32, requires_grad=True)
    # t2 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32, requires_grad=True)
    # t_res = torch.mm(t1, t2)
    # print("result of torch ic carpim: \n", t_res)

    # print("---geri---")

    # # gergen ic carpim
    # print("gergen ic carpim")
    # ones3_3 = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    # g1 = gergen([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # g2 = gergen([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # ic = IcCarpim()
    # res = ic.ileri(g1, g2)
    # print("result of gergen ic carpim: \n", res)
    # back = ic.geri(gergen(ones3_3))
    # print("gradient of the first tensor: \n", back[0])
    # print("gradient of the second tensor: \n", back[1])

    # print("#############################################")

    # # torch ic carpim
    # print("torch ic carpim")
    # t1 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32, requires_grad=True)
    # t2 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32, requires_grad=True)
    # t_res = torch.mm(t1, t2)
    # print("result of torch ic carpim: \n", t_res)
    # grad_tensor = torch.ones_like(t_res)
    # t_res.backward(grad_tensor)
    # print("gradient of the first tensor: \n", t1.grad)
    # print("gradient of the second tensor: \n", t2.grad)

    # test geri of a operation
    # print(gArray.operation.b)
    # print(gArray.operation.geri(gergen(ones))[1])

    #### test torch dis carpim
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

    #### test relu
    # print("-------RELU-------")

    # print("---ileri---")

    # # gergen relu
    # print("gergen relu")
    # g1 = gergen([1, 2, -1])
    # relu = ReLU()
    # res = relu.ileri(g1)
    # print("result of gergen relu: \n", res)

    # print("#############################################")

    # # torch relu
    # print("torch relu")
    # t1 = torch.tensor([1, 2, -1], dtype=torch.float32, requires_grad=True)
    # t_relu = torch.nn.ReLU()
    # t_res = t_relu(t1)
    # print("result of torch relu: \n", t_res)

    # print("---geri---")

    # # gergen relu
    # print("gergen relu")
    # ones3 = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    # g1 = gergen([[1, 2, -1], [-3, 0, 4], [-1, -0.3, 1.5]], operation=ReLU(), requires_grad=True)
    # relu = ReLU()
    # res = relu.ileri(g1)
    # print("result of gergen relu: \n", res)
    # back = relu.geri(gergen(ones3))
    # print("gradient of the tensor: \n", back)

    # print("#############################################")

    # # torch relu
    # print("torch relu")
    # t1 = torch.tensor([[1, 2, -1], [-3, 0, 4], [-1, -0.3, 1.5]], dtype=torch.float32, requires_grad=True)
    # t_relu = torch.nn.ReLU()
    # t_res = t_relu(t1)
    # print("result of torch relu: \n", t_res)
    # grad_tensor = torch.ones_like(t_res)
    # t_res.backward(grad_tensor)
    # print("gradient of the tensor: \n", t1.grad)

    #### test softmax
    # print("-------SOFTMAX-------")

    # print("---ileri---")

    # # gergen softmax
    # print("gergen softmax")
    # g1 = gergen([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # softmax = Softmax()
    # res = softmax.ileri(g1, dim=1)
    # print("result of gergen softmax: \n", res)

    # print("#############################################")

    # # torch softmax
    # print("torch softmax")
    # t1 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float64, requires_grad=True)
    # t_softmax = torch.nn.Softmax(dim=1)
    # t_res = t_softmax(t1)
    # print("result of torch softmax: \n", t_res)

    # print("---geri---")

    # # gergen softmax
    # print("gergen softmax")
    # g1 = gergen([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # ones3_3 = (gergen(g1.custom_zeros(g1.boyut())) + 1).veri
    # softmax = Softmax()
    # res = softmax.ileri(g1, dim=1)
    # print("result of gergen softmax: \n", res)
    # back = softmax.geri(gergen(ones3_3))
    # print("gradient of the tensor: \n", back)

    # print("#############################################")

    # # torch softmax
    # print("torch softmax")
    # t1 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float64, requires_grad=True)
    # t_softmax = torch.nn.Softmax(dim=1)
    # t_res = t_softmax(t1)
    # print("result of torch softmax: \n", t_res)
    # grad_tensor = torch.ones_like(t_res)
    # t_res.backward(grad_tensor)
    # print("gradient of the tensor: \n", t1.grad)

    #### test Katman class
    print("-------KATMAN-------")

    print("---ileri---")

    # gergen katman
    print("gergen katman")
    g1 = gergen([[1], [1], [1]])
    katman = Katman(3, 2, "relu")
    res = katman.ileri(g1)
    print("result of gergen katman: \n", res)


if __name__ == "__main__":
    cekirdek(2)
    # data_file = "./path_to_data"
    # main(data_file)
    main()
