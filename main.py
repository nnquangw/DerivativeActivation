import DerivativeCalculation
import argparse

def activation_function(act, a, x):
    """
    Derivative calculation of activation function
    :param act: activation function's name
    :param a: input factor a
    :param x: input x
    :return: derivation of activation function
    """
    if (act == "identical"):
        return DerivativeCalculation.derivativeIdentical(x)
    elif (act == "logistic"):
        return DerivativeCalculation.derivativeLogistic(x)
    elif (act == "tanh"):
        return DerivativeCalculation.derivativeTanh(x)
    elif (act == "relu"):
        return DerivativeCalculation.derivativeReLU(x)
    elif (act == "prelu"):
        return DerivativeCalculation.derivativePReLU(a, x)
    elif (act == "elu"):
        return DerivativeCalculation.derivativeELU(a, x)
    elif (act == "softplus"):
        return DerivativeCalculation.derivativeSoftPlus(x)

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Derivative Calculation of Activation functions')
    parser.add_argument('activation_name', default='', help='activation function name')
    parser.add_argument('a', type=float, default=0, help='factor a')
    parser.add_argument('x', type=float, default=0, help='x')

    args = parser.parse_args()
    print(activation_function(args.activation_name, args.a, args.x))
