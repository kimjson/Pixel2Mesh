from torch.nn import Module, ReLU, ModuleList
from pytorch3d.ops import GraphConv

from graph_convolution import GraphConvolution

class GResNet(Module) : 
    def __init__(self, inputDim, outputDim):
        super(GResNet, self).__init__()
        
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.gcnlayers = [GraphConv(inputDim, outputDim).to("cuda")]+[GraphConv(outputDim, outputDim).to("cuda") for i in range(12)]+[GraphConv(outputDim, 3).to("cuda")]
        self.gcnlayers = ModuleList(self.gcnlayers)

    def forward(self, edges, shape_features):
        shape_features = self.gcnlayers[0](shape_features, edges)
        shape_features = ReLU()(shape_features)
        for i in range(1, len(self.gcnlayers) - 2, 2):
            layer1 = self.gcnlayers[i]
            layer2 = self.gcnlayers[i+1]
            temp = shape_features
            shape_features = layer1(shape_features, edges)
            shape_features = ReLU()(shape_features)
            shape_features = layer2(shape_features, edges)
            shape_features = ReLU()(shape_features)
            shape_features =(temp + shape_features)/2
        coordinates = self.gcnlayers[-1](shape_features, edges)
        return shape_features, coordinates
