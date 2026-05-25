# Linear Regression allows us to determine and study the relationship between two continuous variables. This is a Pytorch Implementation of Linear regression. 

import torch

""" 
(x1​,y1​)=(1,2)
(x2,y2)=(2,4)
(x3,y3)=(3,6)

""" 

# What Variable(...) is, Variable was old PyTorch syntax from before, 2018. These days, plain tensors do everything Variable used to do.
""" 
Modern equivalent of following code: 

```python 

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])

``` 
"""

x_data = torch.tensor([[1.0], [2.0], [3.0]]) #input data 
y_data = torch.tensor([[2.0], [4.0], [6.0]]) # actual y data, this will be used while claculating loss


""" 
In pytorch there are two main steps associated with defining our model. 
They are: 

1. Initializing our model 
2. Declaring the forward pass 
"""

class LinearRegressionModel(torch.nn.Module):

    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        ''' 
        ``` python 
         self.linear = torch.nn.Linear(1, 1)

        ``` 
        This is the actual model. nn.Linear(1, 1) which creates a linear layer:
            First 1 = input has 1 feature
            Second 1 = output has 1 value

        linear layer --> linear regression model ( y = m x + b , m = weight/slope, b = bias/y-intercept)

        '''
        
        self.linear = torch.nn.Linear(1, 1)  


    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
    

"""
Here select the optimizer and the loss criteria. Here, we will use the mean squared error (MSE) as our loss function and stochastic gradient descent (SGD) as our optimizer. Also, we arbitrarily fix a learning rate of 0.01.
"""

# model
our_model = LinearRegressionModel()

# criterion --> loss, here we are using Mean Square Loss
criterion = torch.nn.MSELoss(reduction = 'sum')

# similar to gradient descent, we are using stochastic gradient descent
optimizer = torch.optim.SGD(our_model.parameters(), lr = 0.01)

'''
This file does AUTOMATICALLY what the notebook (linear_regression.ipynb) does by hand:
    - nn.Linear         holds the w and b we declared manually with torch.tensor(..., requires_grad=True)
    - optimizer.step()  replaces  w.data = w.data - lr * w.grad.data
    - optimizer.zero_grad() replaces  w.grad.data.zero_()
'''



""" 
We now arrive at our training step. We perform the following tasks 500 times during training: 
 

1. Perform a forward pass bypassing our data and finding out the predicted value of y.
2. Compute the loss using MSE.
3. Reset all the gradients to 0, perform a backpropagation and then, update the weights.

"""
for epoch in range(500):

    # Forward pass: Compute predicted y by passing 
    # x to the model
    pred_y = our_model(x_data)

    # Compute and print loss
    loss = criterion(pred_y, y_data)

    # Zero gradients, perform a backward pass, 
    # and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # print every 50 epochs so the output shows the trend without 500 lines of noise
    if epoch % 50 == 0:
        print('epoch {}, loss {}'.format(epoch, loss.item()))


# the true rule was y = 2x, so we expect weight ~ 2 and bias ~ 0
w, b = our_model.linear.weight.item(), our_model.linear.bias.item()
print(f"learned: y = {w:.3f}x + {b:.3f}")



""" 

"""
new_var = torch.tensor([[4.0]])
pred_y = our_model(new_var)
print("predict (after training)", 4, pred_y.item())