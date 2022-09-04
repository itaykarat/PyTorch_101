import torch


class get_info:
    def __init__(self,Primitive_type_input):
        self.Goal = 'Class that shows the info we can get from a tensor object'
        self.tensor = self.create_tensor(Primitive_type_input)

    def create_tensor(self, Primitive_type_input):
        new_tensor = torch.tensor(Primitive_type_input)
        return new_tensor

    def get_shape(self):
        return self.tensor.shape

    def get_dtype(self):
        return self.tensor.dtype

    def get_device_tensor_stored_in(self):
        return self.tensor.device



if __name__ == '__main__':
    Primitive_type_input = [[[1, 2, 3], [3, 6, 9], [2, 4, 5]]]

    tensor_example = get_info(Primitive_type_input)
    print('tensor value -->\n', tensor_example.tensor)
    print('tensor original type -->', type(tensor_example.tensor))
    print('tensor Shape -->', tensor_example.get_shape())
    print(f'tensor Dtype --> {tensor_example.get_dtype()}')
    print(f'Device that tensor stored in --> {tensor_example.get_device_tensor_stored_in()}')




import torch


class get_info:
    def __init__(self,Primitive_type_input):
        self.Goal = 'Class that shows the info we can get from a tensor object'
        self.tensor = self.create_tensor(Primitive_type_input)

    def create_tensor(self, Primitive_type_input):
        new_tensor = torch.tensor(Primitive_type_input)
        return new_tensor

    def get_shape(self):
        return self.tensor.shape

    def get_dtype(self):
        return self.tensor.dtype

    def get_device_tensor_stored_in(self):
        return self.tensor.device



if __name__ == '__main__':
    Primitive_type_input = [[[1, 2, 3], [3, 6, 9], [2, 4, 5]]]

    tensor_example = get_info(Primitive_type_input)
    print('tensor value -->\n', tensor_example.tensor)
    print('tensor original type -->', type(tensor_example.tensor))
    print('tensor Shape -->', tensor_example.get_shape())
    print(f'tensor Dtype --> {tensor_example.get_dtype()}')
    print(f'Device that tensor stored in --> {tensor_example.get_device_tensor_stored_in()}')




