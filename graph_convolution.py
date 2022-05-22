import torch
from torch.nn import Linear, Module

class GraphConvolution(Module): 
    def __init__(self, inputDim, outputDim):
        super().__init__()
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.linear1 = Linear(self.inputDim, self.outputDim)
        self.linear2 = Linear(self.inputDim, self.outputDim)
        
    def forward(self,neighbours,shape_features):
        outputFeatures1 = self.linear1(shape_features) 
        shapeFeaturesAggr = torch.empty(shape_features.size(), device="cuda")
        for index,neighbour in enumerate(neighbours):
            neighbour = torch.tensor(list(neighbour), device="cuda")
            shapeFeaturesAggr[index] = torch.sum(shape_features[neighbour], 0) 
        shape_features = outputFeatures1 + self.linear2(shapeFeaturesAggr)
        return shape_features