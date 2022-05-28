from unittest import result
from pytorch3d.loss import chamfer_distance 
from pytorch3d.ops import knn_points
import torch
def p2m_loss (prediction, g_truth, g_truth_normals, neighbours): 
    chamferloss, chamferloss_normals = chamfer_distance(prediction, g_truth)
    loss = normal_loss(prediction, g_truth, g_truth_normals, neighbours) + chamferloss
    return loss

#TODO : normalize vectors
def normal_loss(prediction, g_truth, g_truth_normals, neighbours):
    nn_indices = knn_points(prediction, g_truth, return_nn = True).idx
    nn_indices = nn_indices[0]
    prediction = prediction[0]
    g_truth_normals = g_truth_normals[0]
    result = 0
    for index, neighbour in enumerate(neighbours) : 
        surface_normal = g_truth_normals[nn_indices[index].item()]
        for vert in neighbour : 
            edge = prediction[vert] - prediction[index]
            result +=torch.dot(edge, surface_normal)
    return result
