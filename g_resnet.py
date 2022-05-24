from graph_convolution import GraphConvolution
from torch.nn import Module, ReLU

class GResNet(Module) : 
    def __init__(self, inputDim, outputDim):
        super().__init__()
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.gcnlayers = [GraphConvolution(inputDim, outputDim).to("cuda")]+[GraphConvolution(outputDim, outputDim).to("cuda") for i in range(13)]
        self.gcnlayers = self.gcnlayers[:1] + [self.gcnlayers[i:i + 2] for i in range(1, 13, 2)] + self.gcnlayers[-1:] 
        self.extra_gcnlayer = GraphConvolution(outputDim, 3).to("cuda")
    def forward(self, neighbours, shape_features):
        shape_features = self.gcnlayers[0](neighbours, shape_features)
        shape_features = ReLU()(shape_features)
        for layer1, layer2 in self.gcnlayers[1:-1]:
            temp = shape_features
            shape_features = layer1(neighbours, shape_features)
            shape_features = ReLU()(shape_features)
            shape_features = layer2(neighbours, shape_features)
            shape_features = ReLU()(shape_features)
            shape_features =(temp + shape_features)/2
        shape_features = self.gcnlayers[-1](neighbours, shape_features)
        shape_features = ReLU()(shape_features)
        coordinates = self.extra_gcnlayer(shape_features)
        return shape_features, coordinates
