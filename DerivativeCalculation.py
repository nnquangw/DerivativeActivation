import math


def derivativeIdentical(x):
    """
    Derivative calculation of Identical activation function
    f(x) = x
    :param x: input x
    :return: derivation of Identical function
    """
    return 1


def derivativeLogistic(x):
    """
    Derivative calculation of Logistic activation function
    f(x) = 1 / (1 + exp(-x))
    :param x: input x
    :return: derivation of Logistic function
    """
    numerator = math.exp(-x)*(-1)
    denominator = -(1 + math.exp(-x))*(1 + math.exp(-x))
    return numerator / denominator


def derivativeTanh(x):
    """
    Derivative calculation of Tanh activation function
    f(x) = tanh(x) = (2 / (1+exp(-2x))) - 1
    :param x: input x
    :return: derivation of Tanh function
    """
    numerator = 2 * math.exp(-2*x) * -2
    denominator = -(1 + math.exp(-2*x))*(1 + math.exp(-2*x))
    return numerator / denominator


def derivativeReLU(x):
    """
    Derivative calculation of Rectified Linear Unit activation function
    f(x) = 0    when x < 0
    f(x) = x    when x >= 0
    :param x: input x
    :return: derivation of ReLU function
    """
    result = (x > 0)*1
    return result


def derivativePReLU(a, x):
    """
    Derivative calculation of Parameteric Rectified Linear Unit activation function
    f(x) = ax   when x<0
    f(x) = x    when x>=0
    :param a: input factor a
    :param x: input x
    :return: derivation of PReLU function
    """
    result = (x < 0)*x*a + (x >= 0)*x
    return result


def derivativeELU(a, x):
    """
    Derivative calculation of Exponential Linear Unit activation function
    f(x) = a*(exp(x)-1) when x<0
    f(x) = x            when x>=0
    :param a: input factor a
    :param x: input x
    :return: derivation of ELU function
    """
    result = (x < 0)*(a*math.exp(x)) + (x >= 0)*x
    return result


def derivativeSoftPlus(x):
    """
    Derivative calculation of SoftPlus activation function
    f(x) = log_e(1+exp(x))
    :param x:
    :return:
    """
    numerator = math.exp(x)
    denominator = (1+math.exp(x))*1
    return numerator / denominator
