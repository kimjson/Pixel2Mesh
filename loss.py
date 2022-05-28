from unittest import result
from pytorch3d.loss import chamfer_distance 
from pytorch3d.ops import knn_points
import torch
def p2m_loss (prediction, g_truth, g_truth_normals, neighbours, laplacian_regularization_value): 
    chamferloss, _ = chamfer_distance(prediction, g_truth)
    loss = normal_loss(prediction, g_truth, g_truth_normals, neighbours) + chamferloss + edge_regularization(prediction, neighbours) + laplacian_regularization_value
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

def edge_regularization(prediction,neighbours):
    result = 0
    prediction = prediction[0]
    for index, neighbour in enumerate(neighbours) : 
        for vert in neighbour : 
            result += torch.norm(prediction[vert] - prediction[index])**2
    return result/2.0
def laplacian_regularization(vertices_before,vertices_after, neighbours) : 
    vertices_before = vertices_before[0]
    vertices_after = vertices_after[0]
    loss = 0
    for index, neighbour in enumerate(neighbours):
        neighbour_indices = torch.tensor(list(neighbour))
        sum_before = torch.sum(vertices_before[neighbour_indices])
        delta_before = vertices_before[index] - sum_before
        sum_after = torch.sum(vertices_after[neighbour_indices])
        delta_after =  vertices_after[index]-sum_after
        loss += torch.norm(delta_after - delta_before)**2
    return loss