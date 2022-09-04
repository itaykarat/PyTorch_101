import torch


class aggregation:
    def __init__(self):
        self.Goal = 'show aggregation functions in PyTorch'
        self.tensor = self.create_tensor()

    def create_tensor(self):
        """
        # create a tensor with elements from 0 to 100 with step of 10 between
        # elements
        # will look like tensor([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        """
        new_tensor = torch.arange(0, 100, 10)
        return new_tensor

    def get_min(self):
        return self.tensor.min()

    def get_max(self):
        return self.tensor.max()

    def get_mean(self):
        """
         # Note that I needed to cast the tensor's type to be able to
         # perform mean op.
        """
        return self.tensor.type(torch.float32).mean()

    def get_sum(self):
        return self.tensor.sum()

    ##### argmin & arg max  #####

    """
    Positional min/max
    Tools to find the index of a tensor where the max or minimum occurs with torch.
    argmax() and torch.argmin() respectively.

    This is helpful in case you just want the position where the highest (or lowest) value is and not the actual value 
    itself (we'll see this in a later section when using the softmax activation function).
     """

    def get_argmax(self):
        return self.tensor.argmax()

    def get_argmin(self):
        return self.tensor.argmin()


if __name__ == '__main__':
    aggregation_obj = aggregation()

    print(f"Minimum: {aggregation_obj.get_min()}")
    print(f"Maximum: {aggregation_obj.get_max()}")
    print(f"Mean: {aggregation_obj.get_mean()}")  # won't work without float datatype
    print(f"Sum: {aggregation_obj.get_sum()}")

    print(f"index of tensor where max in: {aggregation_obj.get_argmax()}")
    print(f"index of tensor where min in: {aggregation_obj.get_argmin()}")
