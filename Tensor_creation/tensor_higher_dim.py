import torch


class Tensor:
    def __init__(self, Primitive_type_input):
        self.Goal = 'Class that shows how to define a Tensor of high dimension in PyTorch'
        self.tensor = self.create_tensor(Primitive_type_input)

    def create_tensor(self, Primitive_type_input):
        new_tensor = torch.tensor(Primitive_type_input)
        return new_tensor

    def return_dim(self):
        return self.tensor.ndim

    def return_shape(self):
        return self.tensor.shape



if __name__ == '__main__':
    Primitive_type_input = [[[1, 2, 3], [3, 6, 9], [2, 4, 5]]]

    tensor_example = Tensor(Primitive_type_input)
    print('tensor value -->\n', tensor_example.tensor)
    print('tensor dim -->', tensor_example.return_dim())
    print('tensor original type -->', type(tensor_example.tensor))
    print('tensor Shape -->', tensor_example.return_shape())



