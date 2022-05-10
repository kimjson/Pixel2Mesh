import torch
import torch.nn as nn
from torchvision.models import vgg16

class P2M(nn.Module):
    def __init__(self):
        super().__init__()

        self.vgg16 = vgg16(pretrained=True)
        self.vgg16.eval()

        self.vgg16_conv3_3_layer = list(self.vgg16.features)[15]
        self.vgg16_conv4_3_layer = list(self.vgg16.features)[22]
        self.vgg16_conv5_3_layer = list(self.vgg16.features)[29]

        def conv3_3_hook(model, input, output):
            self.vgg16_conv3_3_feature = output.detach()

        def conv4_3_hook(model, input, output):
            self.vgg16_conv4_3_feature = output.detach()

        def conv5_3_hook(model, input, output):
            self.vgg16_conv5_3_feature = output.detach()

        self.vgg16_conv3_3_layer.register_forward_hook(conv3_3_hook)
        self.vgg16_conv4_3_layer.register_forward_hook(conv4_3_hook)
        self.vgg16_conv5_3_layer.register_forward_hook(conv5_3_hook)

        # TODO: implement G-ResNet
        self.g_resnet1 = None
        self.g_resnet2 = None
        self.g_resnet3 = None

    # @return 2d coordinates
    def project_2d(self, coordinates, vgg16_features, camera):
        raise NotImplementedError()

    def bilinear_interpolation(self, coordinates_2d, vgg16_features):
        raise NotImplementedError()

    def pool_perception_feature(self, mesh, vgg16_features, camera):
        coordinates, feature = mesh
        coordinates_2d = self.project_2d(coordinates, vgg16_features, camera)
        perception_features = self.bilinear_interpolation(coordinates_2d, vgg16_features)

        # TODO: concat three perception features
        return None


    def generate_initial_mesh(self):
        raise NotImplementedError()

    def unpool_graph(self, graph):
        raise NotImplementedError()

    # TODO: Add more necessary parameters
    def deform_mesh(self, mesh, vgg16_features, camera, g_resnet):
        coordinates, feature = mesh
        perception_feature = self.pool_perception_feature(mesh, vgg16_features, camera)

        # TODO: concat perception_feature with feature
        feature_input = None
        
        # TODO: add another branch to calculate new coordinates
        return g_resnet(feature_input)

    # @param image - 137x137 image
    # @param camera - camera intrinsic and extrinsic matrices
    def forward(self, image, camera):
        # TODO: Feed image into vgg16
        self.vgg16(image)

        vgg16_features = [
            self.vgg16_conv3_3_feature,
            self.vgg16_conv4_3_feature,
            self.vgg16_conv5_3_feature,
        ]

        mesh = self.generate_initial_mesh()

        mesh = self.deform_mesh(mesh, vgg16_features, camera, self.g_resnet1)
        mesh = self.unpool_graph(mesh)

        mesh = self.deform_mesh(mesh, vgg16_features, camera, self.g_resnet2)
        mesh = self.unpool_graph(mesh)

        mesh = self.deform_mesh(mesh, vgg16_features, camera, self.g_resnet3)

        return mesh
