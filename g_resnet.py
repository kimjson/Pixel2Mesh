from graph_convolution import GraphConvolution
from torch.nn import Module, ReLU

class GResNet(Module) : 
    def __init__(self, inputDim, outputDim):
        super().__init__()
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.gcnlayers = [GraphConvolution(inputDim, outputDim).to("cuda")]+[GraphConvolution(outputDim, outputDim).to("cuda") for i in range(13)]
        self.gcnlayers = [self.gcnlayers[i:i + 2] for i in range(0, 14, 2)] 
    def forward(self, mesh, shape_features):
        for layer1, layer2 in self.gcnlayers:
            temp = shape_features
            shape_features = layer1(mesh, shape_features)
            shape_features = ReLU()(shape_features)
            shape_features = layer2(mesh, shape_features)
            shape_features = ReLU()(shape_features)
            shape_features +=temp
        return shape_features
