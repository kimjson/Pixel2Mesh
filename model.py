import torch
import torch.nn as nn
from torchvision.models import vgg16
from scipy.interpolate import interp2d # TODO: add scipy to environment.yml

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

        def create_g_resnet():
            # TODO: implement G-ResNet
            return nn.Identity()

        self.g_resnet1 = create_g_resnet()
        self.g_resnet2 = create_g_resnet()
        self.g_resnet3 = create_g_resnet()

    # @return 2d coordinates, scale [0, 1]
    def project_2d(self, coordinates, vgg16_features, camera_c, camera_f):
        raise NotImplementedError()

    def pool_features(self, coordinates_2d, vgg16_features):
        def bilinear_interpolation(coordinates_2d, vgg16_feature):
            # Get dimension of vgg16 feature
            dimension = vgg16_feature.shape[0]
            # Rescale coordinate with the dimension
            coordinates_2d = coordinates_2d * dimension
            # Call interp2d, TODO: make shape consistent
            return interp2d(coordinates_2d[0], coordinates_2d[1], vgg16_feature)
            
        return [
            bilinear_interpolation(coordinates_2d, vgg16_features[0]),
            bilinear_interpolation(coordinates_2d, vgg16_features[1]),
            bilinear_interpolation(coordinates_2d, vgg16_features[2]),
        ]

    def pool_perception_feature(self, mesh, vgg16_features, camera_c, camera_f):
        coordinates, feature = mesh
        coordinates_2d = self.project_2d(coordinates, vgg16_features, camera_c, camera_f)
        perception_features = self.pool_features(coordinates_2d, vgg16_features)

        # TODO: concat three perception features
        return perception_features

    def generate_initial_mesh(self):
        # TODO: implement
        return None

    def unpool_graph(self, graph):
        raise NotImplementedError()

    def deform_mesh(self, mesh, vgg16_features, camera_c, camera_f, g_resnet):
        # coordinates, feature = mesh
        perception_feature = self.pool_perception_feature(mesh, vgg16_features, camera_c, camera_f)

        # TODO: concat perception_feature with feature
        feature_input = None
        
        # TODO: add another branch to calculate new coordinates
        return g_resnet(feature_input)

    # @param image - 137x137 image
    # @param camera - camera intrinsic and extrinsic matrices
    def forward(self, image, camera_c, camera_f):
        self.vgg16(image)

        vgg16_features = [
            self.vgg16_conv3_3_feature,
            self.vgg16_conv4_3_feature,
            self.vgg16_conv5_3_feature,
        ]

        mesh = self.generate_initial_mesh()

        mesh = self.deform_mesh(mesh, vgg16_features, camera_c, camera_f, self.g_resnet1)
        mesh = self.unpool_graph(mesh)

        mesh = self.deform_mesh(mesh, vgg16_features, camera_c, camera_f, self.g_resnet2)
        mesh = self.unpool_graph(mesh)

        mesh = self.deform_mesh(mesh, vgg16_features, camera_c, camera_f, self.g_resnet3)

        return mesh
