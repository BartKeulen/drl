def tensor_shape_string(tensor):
    """
    Creates a string of the shape of the tensor with format (dim[0], dim[1], ..., dim[n])

    :param tensor: input tensor
    :return: String of shape
    """
    shape = tensor.get_shape().as_list()
    shape_str = '('
    for i in range(len(shape)):
        shape_str += '{:s}'.format(str(shape[i]))
        if i < len(shape)-1:
            shape_str += ', '
    shape_str += ')'
    return shape_str


def tensor_num_params(tensor):
    """
    Returns the number of params in the tensor, can only be done if the size is finite, no dimension of shape None.

    E.g.
        - tensor with shape (50, 30) returns 1500
        - tensor with shape (None, 30) raises exception


    :param tensor: input tensor
    :return: number of params in tensor
    """
    shape = tensor.get_shape().as_list()
    num_params = 1
    for i in range(len(shape)):
        if shape[i] is None:
            raise Exception("Can only calculate number of params when size is fixed, e.g. no tensor with shape [None, 10]")
        num_params *= shape[i]
    return num_params
