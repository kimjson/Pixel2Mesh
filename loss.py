from pytorch3d.loss import chamfer_distance 
def p2m_loss (prediction, g_truth): 
    loss, loss_normals = chamfer_distance(prediction, g_truth)
    return loss 