from torch.nn import Module, ReLU, ModuleList
from torch.nn.init import xavier_uniform_
from pytorch3d.ops import GraphConv

from graph_convolution import GraphConvolution

def create_graph_conv(input_dim, output_dim):
    graph_conv = GraphConv(input_dim, output_dim).to("cuda")
    # xavier_uniform_(graph_conv.w0.weight, gain=0.1)
    # xavier_uniform_(graph_conv.w1.weight, gain=0.1)

    return graph_conv

class GResNet(Module) : 
    def __init__(self, inputDim, outputDim):
        super(GResNet, self).__init__()
        
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.gcnlayers = [create_graph_conv(inputDim, outputDim)]+[create_graph_conv(outputDim, outputDim) for i in range(12)]+[create_graph_conv(outputDim, 3)]
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
