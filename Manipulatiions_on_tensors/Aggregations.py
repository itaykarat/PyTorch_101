import torch

class aggregation:
    def __init__(self):
        self.Goal = 'show aggregation functions in PyTorch'
        self.tensor = self.create_tensor()


    def create_tensor(self):
        new_tensor = torch.arange(0, 100, 10) # create a tensor with elements from 0 to 100 with step of 10 between elements
                                              # will look like tensor([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        return new_tensor


    def get_min(self):
        return self.tensor.min()


    def get_max(self):
        return self.tensor.max()

    def get_mean(self):
        return self.tensor.type(torch.float32).mean()

    def get_sum(self):
        return self.tensor.sum()


if __name__ == '__main__':
    aggregation_obj = aggregation()

    print(f"Minimum: {aggregation_obj.get_min()}")
    print(f"Maximum: {aggregation_obj.get_max()}")
    # print(f"Mean: {aggregation_obj.get_mean()}") # this will error
    print(f"Mean: {aggregation_obj.get_mean()}")  # won't work without float datatype
    print(f"Sum: {aggregation_obj.get_sum()}")