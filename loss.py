from unittest import result
from pytorch3d.loss import chamfer_distance 
from pytorch3d.ops import knn_points
from torch.nn.functional import mse_loss, normalize
import torch
def p2m_loss (prediction, g_truth, g_truth_normals, neighbours, laplacian_regularization_value, move_loss_value): 
    chamferloss, _ = chamfer_distance(prediction, g_truth)

    chamfer_term = chamferloss*3000
    normal_term = normal_loss(prediction, g_truth, g_truth_normals, neighbours)*0.5
    laplacian_term = laplacian_regularization_value*1500
    move_term = move_loss_value*100
    edge_term = edge_regularization(prediction, neighbours)*300
    
    loss = chamfer_term + normal_term + laplacian_term + move_term + edge_term

    print(f'chamfer_term: {chamfer_term * 100 / loss}%')
    print(f'normal_term: {normal_term * 100 / loss}%')
    print(f'laplacian_term: {laplacian_term * 100 / loss}%')
    print(f'move_term: {move_term * 100 / loss}%')
    print(f'edge_term: {edge_term * 100 / loss}%')

    return loss

#TODO : normalize vectors
def normal_loss(prediction, g_truth, g_truth_normals, neighbours):
    nn_indices = knn_points(prediction, g_truth, return_nn = True).idx
    nn_indices = nn_indices[0]
    prediction = prediction[0]
    
    epsilon = 1e-12

    g_truth_normals = normalize(g_truth_normals[0], eps=epsilon)
    result = 0
    max_edge_length = 0
    for index, neighbour in enumerate(neighbours) : 
        surface_normal = g_truth_normals[nn_indices[index].item()]
        for vert in neighbour : 
            edge = prediction[vert] - prediction[index]
            edge_length = edge.norm()
            if edge_length > max_edge_length:
                max_edge_length = edge_length
            result += torch.dot(edge, surface_normal)
    return abs(result / max(epsilon, max_edge_length))

def edge_regularization(prediction,neighbours):
    result = 0
    prediction = prediction[0]
    for index, neighbour in enumerate(neighbours) : 
        for vert in neighbour : 
            result += torch.norm(prediction[vert] - prediction[index])**2
    return result/(2.0*len(neighbours))
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
    return loss / len(neighbours)
def move_loss(vertices_before, vertices_after):
    return mse_loss(vertices_before, vertices_after)
