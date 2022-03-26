import numpy

train_data = (
    ((5, 2, 3), 15),
    ((6, 5, 9), 25),
    ((11, 12, 13), 41),
    ((1, 1, 1), 8),
    ((11, 12, 13), 41),
)

test_data = (
    ((515, 22, 13), 555),
    ((61, 35, 49), 150)
)

parameters = [0, 0, 0, 0]
learning_rate = 0.001


def get_input_value(example_index, parameter_index):
    if parameter_index == 0:
        return 1
    return train_data[example_index][0][parameter_index - 1]


def get_output_value(example_index):
    return train_data[example_index][1]


def get_test_output_value(test_index):
    return test_data[test_index][1]


def hypothesis_value(example_index):  # h(x) for the i-th example case
    val = 0
    for j in range(0, len(parameters)):
        if j == 0:  # the training data starts with x1 not x0
            val += parameters[j]
        else:
            val += parameters[j] * get_input_value(example_index, j)
    return val


def test_hypothesis_value(example_index):  # h(x) for the i-th example case
    val = 0
    for j in range(0, len(parameters)):
        if j == 0:  # the training data starts with x1 not x0
            val += parameters[j]
        else:
            val += parameters[j] * test_data[example_index][0][j - 1]
    return val


def error_value(example_index):
    return hypothesis_value(example_index) - get_output_value(example_index)


def cost_derivative(parameter_index):
    summation = 0
    for i in range(0, len(train_data)):
        summation += error_value(i) * get_input_value(i, parameter_index)
    return summation / len(train_data)


def gradient_descent():
    global parameters
    iterations = 0
    while True:
        iterations += 1
        temp_parameters = [0, 0, 0, 0]
        for j in range(0, len(parameters)):
            temp_parameters[j] = parameters[j] - learning_rate * cost_derivative(j)

        if (numpy.allclose(
                temp_parameters,
                parameters,
                rtol=0,
                atol=0.00000001,
        )):
            break
        parameters = temp_parameters
    print("number of iterations:", iterations)


def test_gradient_descent():
    for i in range(len(test_data)):
        print("\nTest", i + 1, ':')
        print("Real value:", get_test_output_value(i))
        print("Hypothesis value:", test_hypothesis_value(i))


def print_parameters():
    for i in range(0, len(parameters)):
        print('Theta', i, 'equals:', parameters[i])


if __name__ == "__main__":
    gradient_descent()
    print_parameters()
    test_gradient_descent()
