# randonet

> Randomly generate neural network programs

Most neural nets follow the below template in [PyTorch](https://pytorch.org):


```python

import torch.nn as nn

class MyNet(nn.Module):
	def __init__(self, *args, **kwargs): # initialization parameters
		# initialize the nn.Modules that are 
		# part of this network

	def forward(self, inputs):
		# perform a DAG of computations
		# on the inputs and
		return outputs

	## additional methods for data-handling/training/testing/debugging
```

The aim of `randonet` is to try and generate PyTorch modules as per the above template.
Currently it generates only sequential models (check `driver.py`).
