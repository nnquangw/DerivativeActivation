
def derivativeReLU(x):
    """
    Derivative calculation of ReLU activation function
    :param x: input x
    :return: derivation of ReLU function
    """
    result = (x > 0)*1
    return result
