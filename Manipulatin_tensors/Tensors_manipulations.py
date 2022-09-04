import torch
import time


class tensor_manipulations:
    def __init__(self, Primitive_type_input1, Primitive_type_input2):
        self.Goal = 'Class that shows tensor manipulations in PyTorch'
        self.tensor1 = self.create_tensor(Primitive_type_input1)
        self.tensor2 = self.create_tensor(Primitive_type_input2)

    def create_tensor(self, Primitive_type_input):
        new_tensor = torch.tensor(Primitive_type_input)
        return new_tensor

    def addition(self, numeric=True):
        if (numeric):
            return self.tensor1 + self.tensor2
        else:
            return torch.add(self.tensor1, self.tensor2)

    def substraction(self, numeric=True):
        if numeric:
            return self.tensor1 - self.tensor2
        else:
            return torch.sub(self.tensor1, self.tensor2)

    def substraction_scalar_tensor1(self, scalar, numeric=True):
        if numeric:
            return self.tensor1 - scalar
        else:
            return torch.sub(self.tensor1, scalar)

    def substraction_scalar_tensor2(self, scalar, numeric=True):
        if numeric:
            return self.tensor2 - scalar
        else:
            return torch.sub(self.tensor2, scalar)

    def mul_with_scalar_tensor1(self, scalar, numeric=True):
        if numeric:
            return self.tensor1 * scalar
        else:
            return torch.mul(self.tensor1, scalar)

    def mul_with_scalar_tensor2(self, scalar, numeric=True):
        if numeric:
            return self.tensor2 * scalar
        else:
            return torch.mul(self.tensor2, scalar)

    def Multiplication_element_wise(self):
        """
        DIFFERENT FROM MATRIX MUL!
        # Element-wise multiplication (each element multiplies its equivalent, index 0->0, 1->1, 2->2)

        For example:
        tensor([1, 2, 3]) * tensor([1, 2, 3])
        Equals: tensor([1, 4, 9])

        Because --> [1*1,2*2,3*3] = [1,4,9]

        :return: tensor
        """
        return self.tensor1 * self.tensor2

    def Division(self):
        return torch.div(self.tensor1, self.tensor2)

    def Matrix_multiplication(self, numeric=True):
        """
        Reminder : Linear algebra basics.
        Given 2 matrices A1, A2

        Matrix mul will be defined if A1.ROWS = A2.COLS and A1.COLS = A2.ROWS

        for examole:
        A1(ROWS = 3 , COLS = 2) , A2(ROWS = 2 , COLS = 3)

        We denote matrix mul with '@'

        so in code it will look like that:
        A1(3, 2) @ A2(2, 3)
        will work.


        The resulting matrix has the shape of the outer dimensions:
        A1(2, 3) @ A2(3, 2) -> (2, 2)
        A1(3, 2) @ A2(2, 3) -> (3, 3)

        :return: new matrix product of mul with shape of outer dims.
        """
        if numeric:
            return self.tensor1 @ self.tensor2
        else:
            return torch.matmul(self.tensor1, self.tensor2)

    def manual_matmul(self):
        # Matrix multiplication by hand
        # (avoid doing operations with for loops at all cost, they are computationally expensive)
        value = 0
        for i in range(len(self.tensor1)):
            value += self.tensor1[i] * self.tensor1[i]
        print(value)

    def compare_manual_vs_torch_implementation(self):
        start_time = time.time()
        new_mtx = self.manual_matmul()
        print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    """
    Func stack:
    * addition
    * subtraction
    * subtraction_scalar_tensor1
    * subtraction_scalar_tensor2
    * mul_with_scalar_tensor1
    * mul_with_scalar_tensor2
    * Multiplication_element_wise
    * Division
    * Matrix_multiplication
    * manual_matmul
    * compare_manual_vs_torch_implementation
    """

    """
    Deep learning quick peek:
    Neural networks are full of matrix multiplications and dot products!

    The torch.nn.Linear() module (we'll see this in action later on), also known as a feed-forward layer or fully 
    connected layer, implements a matrix multiplication between an input x and a weights matrix A. 
     {Similar to perceptron --> check out my git repo!}
    """

    Primitive_type_input1 = [[[1, 2, 3], [3, 6, 9], [2, 4, 5]]]
    Primitive_type_input2 = [[[8, 5, 4], [3, 6, 9], [5, 4, 3]]]
    tensor_ops = tensor_manipulations(Primitive_type_input1, Primitive_type_input2)

    print('tensor1 value -->\n', tensor_ops.tensor1)
    print('tensor2 value -->\n', tensor_ops.tensor2)

    print('tensor addition -->\n', tensor_ops.addition())
    print('tensor subtraction -->\n', tensor_ops.substraction())
    print('tensor subtraction with scalar tensor1  -->\n', tensor_ops.substraction_scalar_tensor1(3))
    print('tensor subtraction with scalar tensor2  -->\n', tensor_ops.substraction_scalar_tensor2(3))
    print('tensor mul with scalar tensor1  -->\n', tensor_ops.mul_with_scalar_tensor1(2))
    print('tensor mul with scalar tensor2  -->\n', tensor_ops.mul_with_scalar_tensor2(2))
    print('tensors mul element wise  -->\n', tensor_ops.Multiplication_element_wise())
    print('tensors Division  -->\n', tensor_ops.Division())
    print('tensors Matrix_multiplication  -->\n', tensor_ops.Matrix_multiplication())
    print('tensors manual_matmul  -->\n', tensor_ops.manual_matmul())

