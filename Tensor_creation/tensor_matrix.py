import torch

class Matrix:

    def __init__(self,D2_array):
        self.tensor = self.create_matrix(D2_array=D2_array)

    def create_matrix(self,D2_array):
        return torch.tensor(D2_array)


    def return_dim(self):
        return self.tensor.ndim

    def return_shape(self):
        """
        In case of a matrix the shape will be:
        (num_of_rows,num_of_columns)
        :return: tuple represemts the matrix shape
        """
        return self.tensor.shape




if __name__ == '__main__':
    feature_matrix =[[1,2],[3,4],[5,6]]
    matrix_example = Matrix(feature_matrix)
    print('tensor value -->\n', matrix_example.tensor)
    print('tensor dim -->', matrix_example.return_dim())
    print('tensor original type -->', type(matrix_example.tensor))
    print('tensor Shape -->', matrix_example.return_shape())

