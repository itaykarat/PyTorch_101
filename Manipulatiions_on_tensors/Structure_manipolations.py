import torch


class structure_manipolations:
    def __init__(self):
        self.Goal = 'Class that shows tensor structure manipolations'
        self.tensor = self.create_tensor()

    def create_tensor(self):
        x = torch.arange(1., 8.)
        print(f'created new tensor \n {x}\n')
        return x

    def reshape(self):
        reshaped_tensor = self.tensor.reshape(1, 7)
        self.tensor = reshaped_tensor  # setter for the class object
        return reshaped_tensor

    def view(self):
        """
        reshapes the tensor without copying memory, similar to numpy reshape().
        """
        reshaped_tensor = self.tensor.view(1, 7)
        self.tensor = reshaped_tensor  # setter for the class object
        return reshaped_tensor

    def stack(self):
        # Stack tensors on top of each other
        x = self.tensor
        x_stacked = torch.stack([x, x, x, x], dim=0)  # try changing dim to dim=1 and see what happens
        self.tensor = x_stacked
        return x_stacked

    def squeeze(self):
        """
            Decrease tensors dimension
            In order to use dim must be > 1
        """
        squeezed_tensor = self.tensor.squeeze()
        self.tensor = squeezed_tensor
        return squeezed_tensor

    def unsqueeze(self):
        """
            Increase tensors dimension
        """
        tensor_unsqueezed = self.tensor.unsqueeze(dim=0)
        self.tensor = tensor_unsqueezed
        return tensor_unsqueezed

    def permute(self):
        pass


if __name__ == '__main__':
    class_obj = structure_manipolations()

    """RESHAPES THE TENSOR"""
    # print(class_obj.reshape())
    print(class_obj.view())
    print('\n\n')

    """Combine tensots --> AKA stack tensors"""
    print(f'stack the same tensor 4 times\n {class_obj.stack()}\n')  # here I stacked 4 times the tensor's class

    """Squeeze tensors reduce dimensionality by 1"""
    print(f'squeeze the current tensor (dim-1) \n {class_obj.squeeze()}\n')

    """UnSqueeze tensors Increase dimensionality by 1"""
    print(f'unsqueeze the current tensor (dim+1) \n {class_obj.unsqueeze()}\n')
