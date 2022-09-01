import torch


class Scalar:

    def __init__(self,number):
        self.Goal = 'Class that shows how to define a scalar in PyTorch'
        self.tensor = self.create_scalar(number)

    def create_scalar(self,number):
        # Scalar
        scalar = torch.tensor(number)
        return scalar

    def return_dim(self):
        return self.tensor.ndim

    def return_primitive_type_number_from_tensor(self):
        return self.tensor.item()




if __name__ == '__main__':
    dim_0_tensor = Scalar(12)
    print('tensor value -->' , dim_0_tensor.tensor)
    print('tensor dim -->' , dim_0_tensor.return_dim())
    print('tensor original type -->' , type(dim_0_tensor.tensor))
    print('tensor to primitive type -->' , dim_0_tensor.return_primitive_type_number_from_tensor(),type(dim_0_tensor.return_primitive_type_number_from_tensor()))

