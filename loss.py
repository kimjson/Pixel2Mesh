from unittest import result
from pytorch3d.loss import chamfer_distance 
from pytorch3d.ops import knn_points
from torch.nn.functional import mse_loss, normalize
import torch
from torch.profiler import record_function

def p2m_loss(prediction, g_truth, g_truth_normals, adjacency_matrix, laplacian_regularization_value, move_loss_value, edges, is_logging = False): 
    chamferloss, _ = chamfer_distance(prediction, g_truth)

    chamfer_term = chamferloss*3000
    normal_term = normal_loss(prediction, g_truth, g_truth_normals, adjacency_matrix, edges)*0.5
    laplacian_term = laplacian_regularization_value*1500
    move_term = move_loss_value*100
    edge_term = edge_regularization(prediction, adjacency_matrix)*300
    
    loss = chamfer_term + normal_term + laplacian_term + move_term + edge_term

    if is_logging:
        print(f'chamfer_term: {chamfer_term * 100 / loss}%')
        print(f'normal_term: {normal_term * 100 / loss}%')
        print(f'laplacian_term: {laplacian_term * 100 / loss}%')
        print(f'move_term: {move_term * 100 / loss}%')
        print(f'edge_term: {edge_term * 100 / loss}%')

    return loss

def normal_loss(prediction, g_truth, g_truth_normals, adjacency_matrix, edges):
    nn_indices = knn_points(prediction, g_truth, return_nn = True).idx
    nn_indices = nn_indices[0]
    prediction = prediction[0]
    
    epsilon = 1e-12

    g_truth_normals = normalize(g_truth_normals[0], eps=epsilon)
    result = 0
    max_edge_length_value = max_edge_length(prediction, edges)

    for index, neighbour in enumerate(adjacency_matrix) : 
        neighbour_vertices = prediction[neighbour]
        neighbor_edges = neighbour_vertices - prediction[index].repeat(neighbour_vertices.size(0), 1)
        surface_normal = g_truth_normals[nn_indices[index]]

        result = (neighbor_edges @ surface_normal.T).sum()

    return abs(result / max(epsilon, max_edge_length_value))


def max_edge_length(vertices, edges):
    from_vertices = vertices[edges[:, 0]]
    to_vertices = vertices[edges[:, 1]]
    edge_vectors = from_vertices - to_vertices
    return (edge_vectors.norm(dim=0)).max()

def edge_regularization(prediction, edges):
    from_vertices = prediction[0][edges[:, 0]]
    to_vertices = prediction[0][edges[:, 1]]
    edge_vectors = from_vertices - to_vertices
    return (edge_vectors.norm(dim=0) ** 2).sum() / prediction.size(0)

def laplacian_regularization(vertices_before, vertices_after, adjacency_matrix):
    vertices_before = vertices_before[0]
    vertices_after = vertices_after[0]
    loss = 0
    for index, neighbour in enumerate(adjacency_matrix):
        sum_before = torch.sum(vertices_before[neighbour])
        delta_before = vertices_before[index] - sum_before
        sum_after = torch.sum(vertices_after[neighbour])
        delta_after =  vertices_after[index]-sum_after
        loss += torch.norm(delta_after - delta_before)**2
    return loss / adjacency_matrix.size(0)
def move_loss(vertices_before, vertices_after):
    return mse_loss(vertices_before, vertices_after)
