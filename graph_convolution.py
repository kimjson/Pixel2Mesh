import torch
from torch.nn import Linear, Module

class GraphConvolution(Module): 
    def __init__(self, inputDim, outputDim):
        super(GraphConvolution, self).__init__()

        self.inputDim = inputDim
        self.outputDim = outputDim
        
        self.linear1 = Linear(self.inputDim, self.outputDim)
        torch.nn.init.constant_(self.linear1.weight, 0.0)
        torch.nn.init.constant_(self.linear1.bias, 0.0)

        self.linear2 = Linear(self.inputDim, self.outputDim)
        torch.nn.init.constant_(self.linear2.weight, 0.0)
        torch.nn.init.constant_(self.linear2.bias, 0.0)

    def forward(self,neighbours,shape_features):
        print(f"# of non zero weights of linear1: {torch.count_nonzero(list(self.linear1.parameters())[0].data)}")
        print(f"# of non zero weights of linear2: {torch.count_nonzero(list(self.linear2.parameters())[0].data)}")
        outputFeatures1 = self.linear1(shape_features) 
        shapeFeaturesAggr = torch.empty(shape_features.size(), device="cuda")
        for index,neighbour in enumerate(neighbours):
            neighbour = torch.tensor(list(neighbour), device="cuda")
            shapeFeaturesAggr[index] = torch.sum(shape_features[neighbour], 0)
        shape_features = outputFeatures1 + self.linear2(shapeFeaturesAggr)
        return shape_features