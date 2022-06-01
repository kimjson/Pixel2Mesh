import torch
from torch.nn import Linear, Module
from torch.profiler import record_function

class GraphConvolution(Module):
    def __init__(self, inputDim, outputDim):
        super(GraphConvolution, self).__init__()

        self.inputDim = inputDim
        self.outputDim = outputDim

        self.linear1 = Linear(self.inputDim, self.outputDim)
        torch.nn.init.xavier_uniform_(self.linear1.weight, gain=0.1)
        torch.nn.init.constant_(self.linear1.bias, 0.0)

        self.linear2 = Linear(self.inputDim, self.outputDim)
        torch.nn.init.xavier_uniform_(self.linear2.weight, gain=0.1)
        torch.nn.init.constant_(self.linear2.bias, 0.0)

    def forward(self,adjacency_matrix,shape_features):
        with record_function("GraphConvolution.forward"):
            shapeFeaturesAggr = torch.empty(shape_features.size(), device="cuda")
            for index, neighbour in enumerate(adjacency_matrix):
                shapeFeaturesAggr[index] = torch.sum(shape_features[neighbour], 0)
            return self.linear1(shape_features) + self.linear2(shapeFeaturesAggr)