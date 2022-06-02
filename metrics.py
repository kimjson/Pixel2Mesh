from pytorch3d.ops import knn_points
import torch
from emd import earth_mover_distance

def f_score(prediction,g_truth, tau= 1e-4):
    true_positive = knn_points(prediction, g_truth).dists
    true_positive = torch.count_nonzero(true_positive<tau)
    false_positive = prediction.size()[1] - true_positive

    false_negative = knn_points(g_truth, prediction).dists
    false_negative = g_truth.size()[1] - torch.count_nonzero(false_negative<tau)

    precision = true_positive/(false_positive+true_positive)
    recall = true_positive/(false_negative + true_positive)
    f1_score = (recall*precision*2/(recall+precision)).item() if (precision + recall) > 0 else 0
    return f1_score

def emd(prediction, g_truth) :
    emd_value = earth_mover_distance(prediction, g_truth, transpose=False)
    return emd_value