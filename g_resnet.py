from graph_convolution import GraphConvolution
from torch.nn import Module, ReLU, ModuleList, Identity

class GResNet(Module) : 
    def __init__(self, inputDim, outputDim):
        super(GResNet, self).__init__()
        
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.gcnlayers = [GraphConvolution(inputDim, outputDim).to("cuda")]+[GraphConvolution(outputDim, outputDim).to("cuda") for i in range(12)]+[GraphConvolution(outputDim, 3, activation=Identity()).to("cuda")]
        self.gcnlayers = ModuleList(self.gcnlayers)

    def forward(self, neighbours, shape_features):
        shape_features = self.gcnlayers[0](neighbours, shape_features)
        for i in range(1, len(self.gcnlayers) - 2, 2):
            layer1 = self.gcnlayers[i]
            layer2 = self.gcnlayers[i+1]
            temp = shape_features
            shape_features = layer1(neighbours, shape_features)
            shape_features = layer2(neighbours, shape_features)
            shape_features =(temp + shape_features)/2

        coordinates = self.gcnlayers[-1](neighbours, shape_features)

        return shape_features, coordinates
