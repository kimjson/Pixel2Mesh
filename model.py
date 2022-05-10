import torch
import torch.nn as nn
from torchvision.models import vgg16

class P2M(nn.Module):
    def __init__(self):
        super().__init__()

        self.vgg16 = vgg16(pretrained=True)
        self.vgg16_conv3_3 = list(self.vgg16.features)[15]
        self.vgg16_conv4_3 = list(self.vgg16.features)[22]
        self.vgg16_conv5_3 = list(self.vgg16.features)[29]

    def pool_perception_feature(self, img, vertex):
        raise NotImplementedError()

    def generate_initial_mesh(self):
        raise NotImplementedError()

    def unpool_graph(self, graph):
        raise NotImplementedError()

    # TODO: Add more necessary parameters
    def deform_mesh(self):
        raise NotImplementedError()

    def forward(self, image):
        # TODO: Feed image into vgg16

        # TODO: Generate initial mesh
        mesh = self.generate_initial_mesh()

        # TODO: Mesh Deformation
        mesh = self.deform_mesh(mesh)

        # TODO: Graph unpooling
        mesh = self.unpool_graph(mesh)

        # TODO: Mesh Deformation
        mesh = self.deform_mesh(mesh)

        # TODO: Graph unpooling
        mesh = self.unpool_graph(mesh)

        # TODO: Mesh Deformation
        mesh = self.deform_mesh(mesh)

        return mesh
