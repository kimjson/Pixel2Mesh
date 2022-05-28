from pytorch3d.loss import chamfer_distance 
from pytorch3d.ops import knn_points
import torch
def p2m_loss (prediction, g_truth, neighbours): 
    chamferloss, chamferloss_normals = chamfer_distance(prediction, g_truth)
    loss = normal_loss(prediction, g_truth, neighbours) + chamferloss
    return loss

def normal_loss(prediction, g_truth, neighbours):
    knn = knn_points(prediction, g_truth, return_nn = True).knn
    closes_vertices = torch.reshape(knn, (1,10,3))
    edges = torch.empty().cuda()
    for neighbour in neighbours : 

