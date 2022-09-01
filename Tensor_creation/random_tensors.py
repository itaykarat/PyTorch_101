import torch


"""
Random tensors  
We've established tensors represent some form of data.

And machine learning models such as neural networks manipulate and seek patterns within tensors.

But when building machine learning models with PyTorch, it's rare you'll create tenors by hand (like what we've being 
doing). 

Instead, a machine learning model often starts out with large random tensors of numbers and adjusts these random numbers
 
as it works through data to better represent it.

In essence:
Start with random numbers -> look at data -> update random numbers -> look at data -> update random numbers...

As a data scientist, you can define how the machine learning model starts (initialization), 
looks at data (representation) and updates (optimization) its random numbers. 



"""


# global variables to define (size == shape) of the random tensor

class Random_tensor:
    def __init__(self,ROWS=2,COLS=3): # set a default value if parameters not passed in constructor
        self.Goal = 'Class that shows how to define a Tensor of high dimension in PyTorch with random initialization'
        self.rows = ROWS
        self.cols = COLS
        self.tensor = self.create_random_tensor()

    def create_random_tensor(self):
        size = (self.rows,self.cols) # In order to initialize a random vars tensor, Size parameter is needed, size Type: Tuple
        new_tensor = torch.rand(size)
        return new_tensor

    def return_dim(self):
        return self.tensor.ndim


    def return_shape(self):
        return self.tensor.shape



if __name__ == '__main__':

    random_tensor_example = Random_tensor(ROWS=3,COLS=2)
    print('tensor value -->\n', random_tensor_example.tensor)
    print('tensor dim -->', random_tensor_example.return_dim())
    print('tensor original type -->', type(random_tensor_example.tensor))
    print('tensor Shape -->', random_tensor_example.return_shape())

