from turtle import Turtle
from numpy import NaN
import torch
import torch.nn as nn
from torchvision.models import vgg16
from pytorch3d.io.ply_io import load_ply
from pytorch3d.structures import Meshes
from pytorch3d.ops import SubdivideMeshes
from torch.profiler import record_function

from g_resnet import GResNet
from loss import p2m_loss, laplacian_regularization, move_loss

class P2M(nn.Module):
    def __init__(self, ellipsoid_path, camera_c, camera_f):
        super(P2M, self).__init__()

        self.ellipsoid_path = ellipsoid_path
        self.is_train = True
        self.vgg16 = vgg16(pretrained=True)

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

        self.g_resnet1 = GResNet(1283, 128).to("cuda")
        self.g_resnet2 = GResNet(1408, 128).to("cuda")
        self.g_resnet3 = GResNet(1408, 128).to("cuda")

        self.subdivide_meshes = SubdivideMeshes()

        self.camera_c = camera_c
        self.camera_f = camera_f

    # @return pixel coordinates in 224x224 input image
    def image_project(self, coordinates, vgg16_features, camera_c, camera_f):
        projection_matrix = torch.mm(
            torch.tensor([
                [camera_f[0], 0, camera_c[0]],
                [0, camera_f[1], camera_c[1]],
                [0, 0, 1],
            ], device=torch.device('cuda')),
            torch.tensor([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0.8],
            ], device=torch.device('cuda'))
        )
        projection_matrix = projection_matrix.repeat(coordinates.shape[0], 1, 1)

        coordinates = torch.concat([
            coordinates,
            torch.ones(coordinates.shape[0], 1, device=torch.device('cuda'))
        ], 1)
        coordinates = torch.reshape(coordinates, (coordinates.shape[0], coordinates.shape[1], 1))

        pixel_coordinates = torch.bmm(projection_matrix, coordinates)
        pixel_coordinates = torch.reshape(pixel_coordinates, (pixel_coordinates.shape[0], pixel_coordinates.shape[1]))

        s_values = torch.reshape(pixel_coordinates[:, 2], (pixel_coordinates.shape[0], 1)).repeat(1, 3)

        pixel_coordinates = pixel_coordinates / s_values

        pixel_coordinates = pixel_coordinates[:, :-1]

        return pixel_coordinates

    def bilinear_interpolation(self, x, y, feature, image_size):
        dimension = feature.shape[1]
        x = x * dimension / image_size
        y = y * dimension / image_size

        x = torch.clamp(x, min=0, max=dimension - 1)
        y = torch.clamp(y, min=0, max=dimension - 1)

        x1 = torch.floor(x)
        x2 = torch.ceil(x)
        y1 = torch.floor(y)
        y2 = torch.ceil(y)

        feature11 = feature[:, x1.to(torch.long), y1.to(torch.long)].transpose(0, 1)
        feature12 = feature[:, x1.to(torch.long), y2.to(torch.long)].transpose(0, 1)
        feature21 = feature[:, x2.to(torch.long), y1.to(torch.long)].transpose(0, 1)
        feature22 = feature[:, x2.to(torch.long), y2.to(torch.long)].transpose(0, 1)

        w11 = (x2 - x) * (y2 - y)
        w12 = (x2 - x) * (y - y1)
        w21 = (x - x1) * (y2 - y)
        w22 = (x - x1) * (y - y1)

        result = torch.reshape(w11.broadcast_to(feature11.shape[:2]), feature11.shape) * feature11
        result += torch.reshape(w12.broadcast_to(feature12.shape[:2]), feature12.shape) * feature12
        result += torch.reshape(w21.broadcast_to(feature21.shape[:2]), feature21.shape) * feature21
        result += torch.reshape(w22.broadcast_to(feature22.shape[:2]), feature22.shape) * feature22

        return result

    def pool_features(self, coordinates_2d, vgg16_features, image_size):
        x, y = torch.split(coordinates_2d, 1, 1)

        return [
            self.bilinear_interpolation(x, y, vgg16_features[0][0], image_size),
            self.bilinear_interpolation(x, y, vgg16_features[1][0], image_size),
            self.bilinear_interpolation(x, y, vgg16_features[2][0], image_size),
        ]

    def pool_perception_feature(self, mesh, vgg16_features, camera_c, camera_f, image_size):
        coordinates = mesh.verts_list()[0]
        pixel_coordinates = self.image_project(coordinates, vgg16_features, camera_c, camera_f)
        perception_features = self.pool_features(pixel_coordinates, vgg16_features, image_size)

        result = torch.concat(perception_features, 1)
        result = torch.reshape(result, result.shape[:2])

        return result

    def generate_initial_mesh(self):
        vertices, faces = load_ply(self.ellipsoid_path)
        return Meshes(verts=[vertices], faces=[faces]).cuda()

    def unpool_graph(self, graph, shape_features):
        with record_function("P2M.unpool_graph"):
            faces = graph.faces_list()[0]
            vertices = graph.verts_list()[0]
            vertices = torch.cat([vertices, shape_features], 1)
            newFaces = torch.tensor([], device=torch.device('cuda'))
            num_vertices = vertices.shape[0]
            vertex_table = -torch.ones([num_vertices, num_vertices], dtype=torch.long)

            for face in faces :
                i1,i2,i3 = face
                v1 = vertices[i1]
                v2 = vertices[i2]
                v3 = vertices[i3]

                i4 = vertex_table[i1, i2]
                i5 = vertex_table[i2, i3]
                i6 = vertex_table[i3, i1]
                v4 = (v1 + v2)/2
                v5 = (v2 + v3)/2
                v6 = (v3 + v1)/2

                if i4 == -1:
                    vertices = torch.cat([vertices, torch.unsqueeze(v4,0)],0)
                    i4 = vertices.shape[0] - 1
                    vertex_table[i1, i2] = i4
                    vertex_table[i2, i1] = i4

                if i5 == -1:
                    vertices = torch.cat([vertices, torch.unsqueeze(v5,0)],0)
                    i5 = vertices.shape[0] - 1
                    vertex_table[i2, i3] = i5
                    vertex_table[i3, i2] = i5

                if i6 == -1:
                    vertices = torch.cat([vertices, torch.unsqueeze(v6,0)],0)
                    i6 = vertices.shape[0] - 1
                    vertex_table[i3, i1] = i6
                    vertex_table[i1, i3] = i6

                newFaces = torch.cat((newFaces, torch.tensor([[i1,i4,i6]], device=torch.device('cuda'))),0)
                newFaces = torch.cat((newFaces, torch.tensor([[i2,i4,i5]], device=torch.device('cuda'))),0)
                newFaces = torch.cat((newFaces, torch.tensor([[i3,i5,i6]], device=torch.device('cuda'))),0)
                newFaces = torch.cat((newFaces, torch.tensor([[i5,i4,i6]], device=torch.device('cuda'))),0)
                
            shape_features = vertices[:, 3:]
            vertices = vertices[:, :3]
            
            return Meshes(verts=[vertices], faces=[newFaces]).cuda(), shape_features

    def deform_mesh(self, mesh, shape_features, vgg16_features, camera_c, camera_f, g_resnet, image_size, g_truth, g_truth_normals, is_first= False):
        perception_feature = self.pool_perception_feature(mesh, vgg16_features, camera_c, camera_f, image_size)
        features = torch.concat([perception_feature, shape_features], 1)
        
        vertices = mesh.verts_list()[0]
        faces = mesh.faces_list()[0]
        edges = mesh.edges_packed()

        adjacency_matrix = torch.zeros((vertices.shape[0], vertices.shape[0]), device=torch.device("cuda"))
        adjacency_matrix[edges[:, 0], edges[:, 1]] = 1
        adjacency_matrix[edges[:, 1], edges[:, 0]] = 1
        adjacency_matrix = adjacency_matrix.to(torch.bool)
        
        new_features, coordinates = g_resnet(edges, features)
        deformed_mesh = Meshes(verts=[coordinates], faces=[faces]).cuda()
        loss = None
        if self.is_train : 
            vertices_before = torch.unsqueeze(vertices, 0)
            vertices_after = torch.unsqueeze(coordinates, 0)
            laplacian_regularization_value = laplacian_regularization(vertices_before, vertices_after, adjacency_matrix)
            move_loss_value = 0
            if  is_first : 
                laplacian_regularization_value*=0.1
            else : 
                move_loss_value += move_loss(vertices_before, vertices_after)
            loss = p2m_loss(vertices_after, g_truth,g_truth_normals, adjacency_matrix, laplacian_regularization_value, move_loss_value, edges)
        return deformed_mesh, new_features, loss

    def forward(self, image, g_truth, g_truth_normals):
        camera_c = self.camera_c
        camera_f = self.camera_f

        _, __, image_size, ___ = image.shape
        self.vgg16.eval()
        self.vgg16(image)

        vgg16_features = [
            self.vgg16_conv3_3_feature,
            self.vgg16_conv4_3_feature,
            self.vgg16_conv5_3_feature,
        ]

        mesh = self.generate_initial_mesh()

        # Intial shape features are just coordinates (dimension 3)
        shape_features = mesh.verts_list()[0]

        mesh, shape_features, loss_1 = self.deform_mesh(mesh, shape_features, vgg16_features, camera_c, camera_f, self.g_resnet1, image_size, g_truth, g_truth_normals, is_first = True)
        mesh, shape_features = self.subdivide_meshes(mesh, shape_features)

        mesh, shape_features, loss_2 = self.deform_mesh(mesh, shape_features, vgg16_features, camera_c, camera_f, self.g_resnet2, image_size, g_truth, g_truth_normals)
        mesh, shape_features = self.subdivide_meshes(mesh, shape_features)

        mesh, _, loss_3 = self.deform_mesh(mesh, shape_features, vgg16_features, camera_c, camera_f, self.g_resnet3, image_size, g_truth, g_truth_normals)

        loss = (loss_1 + loss_2  + loss_3) if self.is_train else None

        return mesh, loss
