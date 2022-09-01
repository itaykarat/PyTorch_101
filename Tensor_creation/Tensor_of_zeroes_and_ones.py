import torch

class tensor_of_zeroes_or_ones:

    def __init__(self,ROWS=2,COLS=3,ONES = True): # set a default value if parameters not passed in constructor
        self.Goal = 'Class that shows how to define a Tensor of high dimension in PyTorch with ONES OR ZEROES'
        self.rows = ROWS
        self.cols = COLS
        if ONES == True:
            self.tensor = self.create_ones_tensor()
        else:
            self.tensor = self.create_zeroes_tensor()


    def create_zeroes_tensor(self):
        size = (self.rows,self.cols)
        new_tensor = torch.zeros(size)
        return new_tensor

    def create_ones_tensor(self):
        size = (self.rows, self.cols)
        new_tensor = torch.ones(size)
        return new_tensor

    def return_dim(self):
        return self.tensor.ndim

    def return_shape(self):
        return self.tensor.shape


if __name__ == '__main__':
    # Tensor of ONES example
    zeros_or_ones_tensor_example = tensor_of_zeroes_or_ones(ROWS=3,COLS=2,ONES=True)
    print('tensor value -->\n', zeros_or_ones_tensor_example.tensor)
    print('tensor dim -->', zeros_or_ones_tensor_example.return_dim())
    print('tensor original type -->', type(zeros_or_ones_tensor_example.tensor))
    print('tensor Shape -->', zeros_or_ones_tensor_example.return_shape())


    # Tensor of ZEROS example
    zeros_or_ones_tensor_example = tensor_of_zeroes_or_ones(ROWS=3,COLS=2,ONES=False)
    print('tensor value -->\n', zeros_or_ones_tensor_example.tensor)
    print('tensor dim -->', zeros_or_ones_tensor_example.return_dim())
    print('tensor original type -->', type(zeros_or_ones_tensor_example.tensor))
    print('tensor Shape -->', zeros_or_ones_tensor_example.return_shape())
