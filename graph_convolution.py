import torch
from torch.nn import Linear, Module

class GraphConvolution(Module): 
    def __init__(self, inputDim, outputDim):
        super().__init__()
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.linear1 = Linear(self.inputDim, self.outputDim)
        self.linear2 = Linear(self.inputDim, self.outputDim)
        
    def forward(self,mesh,shape_features):
        outputFeatures1 = self.linear1(shape_features) 
        faces = mesh.faces_list()[0]
        vertices = mesh.verts_list()[0]
        neighbours = [set() for i in range(vertices.size()[0])]
        # TODO : migrate neighbours logic outside of forward
        for face in faces : 
            i1,i2,i3 = face
            neighbours[i1] = neighbours[i1].union({i2,i3})
            neighbours[i2] = neighbours[i2].union({i1,i3})
            neighbours[i3] = neighbours[i3].union({i2,i1})
        shapeFeaturesAggr = torch.empty(shape_features.size(), device="cuda")
        for index,neighbour in enumerate(neighbours):
            neighbour = torch.tensor(list(neighbour), device="cuda")
            shapeFeaturesAggr[index] = torch.sum(shape_features[neighbour], 0) 
        shape_features = outputFeatures1 + self.linear2(shapeFeaturesAggr)
        return shape_features