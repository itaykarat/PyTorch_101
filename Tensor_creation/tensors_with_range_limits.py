import torch


class Range_limit_tensor:
    def __init__(self, start=1, end=10, rows=2, columns=3):
        self.Goal = 'Class that shows how to define a Tensor of high dimension in PyTorch with value limit'
        self.rows = rows
        self.columns = columns
        self.size = self.zip_size()
        self.start = start
        self.end = end
        self.tensor = self.create_tensor_with_value_limit()

    def zip_size(self):
        return (self.rows,self.columns)

    def create_tensor_with_value_limit(self):
        new_tensor = torch.randint(low = self.start,high= self.end, size=self.size)
        return new_tensor

    def return_dim(self):
        return self.tensor.ndim

    def return_shape(self):
        return self.tensor.shape


if __name__ == '__main__':
    value_limited_tensor = Range_limit_tensor(start=1, end=10)
    print('tensor value -->\n', value_limited_tensor.tensor)
    print('tensor dim -->', value_limited_tensor.return_dim())
    print('tensor original type -->', type(value_limited_tensor.tensor))
    print('tensor Shape -->', value_limited_tensor.return_shape())
