import torch


class Vector:
    def __init__(self, list):
        self.Goal = 'Class that shows how to define a Vector in PyTorch'
        self.tensor = self.create_vector(list)

    def create_vector(self, list):
        vector = torch.tensor(list)
        return vector

    def return_dim(self):
        return self.tensor.ndim

    # Another important concept for tensors is their shape attribute.
    # The shape tells you how the elements inside them are arranged.
    def return_shape(self):
        """The above returns torch.Size([20]) which means our vector has a shape of [20].
        This is because of the twenty elements we placed inside the square brackets ([x1,...,x20]).
        Let's now see a matrix.

        """
        return self.tensor.shape


if __name__ == '__main__':
    feature_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    vevtor_example = Vector(feature_list)
    print('tensor value -->', vevtor_example.tensor)
    print('tensor dim -->', vevtor_example.return_dim())
    print('tensor original type -->', type(vevtor_example.tensor))
    print('tensor Shape -->', vevtor_example.return_shape())

