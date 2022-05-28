from unittest import result
from pytorch3d.loss import chamfer_distance 
from pytorch3d.ops import knn_points
import torch
def p2m_loss (prediction, g_truth, neighbours): 
    chamferloss, chamferloss_normals = chamfer_distance(prediction, g_truth)
    loss = normal_loss(prediction, g_truth, neighbours) + chamferloss
    return loss

#TODO : normalize vectors
def normal_loss(prediction, g_truth, neighbours, surface_normals):
    nn_indices = knn_points(prediction, g_truth, return_nn = True).idx
    result = 0
    for index, neighbour in enumerate(neighbours) : 
        norm = surface_normals[nn_indices[index]]
        for vert in neighbour : 
            edge = prediction[vert] - prediction[index]
            result +=torch.dot(edge, norm)
    return result
