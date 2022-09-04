import torch


class MatMulDL:
    def __init__(self):
        # Shapes need to be in the right way
        self.tensor_A = torch.tensor([[1, 2],
                                      [3, 4],
                                      [5, 6]], dtype=torch.float32)

        self.tensor_B = torch.tensor([[7, 10],
                                      [8, 11],
                                      [9, 12]], dtype=torch.float32)

    def use_mat_mul_in_neural_net(self):

        # Since the linear layer starts with a random weights matrix, let's make it reproducible (more on this later)
        torch.manual_seed(42)

        # This uses matrix multiplication
        linear = torch.nn.Linear(in_features=2,  # in_features = matches inner dimension of input
                                 out_features=6)  # out_features = describes outer value
        x = self.tensor_A
        output = linear(x)
        print(f'Input tensor:\n {x}\n')
        print(f"Input shape: {x.shape}\n")
        print(f"Output:\n{output}\n\nOutput shape: {output.shape}")


if __name__ == '__main__':
    mat_mul_obj = MatMulDL()  # call the constructor and create 2 tensors
    mat_mul_obj.use_mat_mul_in_neural_net()  # perform matrix mul in a neural net
