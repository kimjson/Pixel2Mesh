from graph_convolution import GraphConvolution
from torch.nn import Module, ReLU

class GResNet(Module) : 
    def __init__(self, inputDim, outputDim, numOfLayers):
        super().__init__()
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.gcnlayers = [GraphConvolution(inputDim, outputDim).to("cuda")]+[GraphConvolution(outputDim, outputDim).to("cuda") for i in range(numOfLayers - 1)]
        
    def forward(self, mesh, shape_features):
        for layer in self.gcnlayers:
            relu = ReLU()
            shape_features = layer(mesh, shape_features)
            shape_features = relu(shape_features)

        return shape_features
