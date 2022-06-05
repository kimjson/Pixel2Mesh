import datetime
import traceback
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.contrib import tenumerate
from pytorch3d.loss import chamfer_distance

from metrics import emd, f_score
from dataset import ShapeNet
from model import P2M

device = "cuda" if torch.cuda.is_available() else "cpu"

# TODO: replace with command line parameters
ellipsoid_path = "./data/initial_ellipsoid.ply"
camera_c = [112.0, 112.0]
camera_f = [250.0, 250.0]

@torch.no_grad()
def test(dataloader, model):
    model.eval()
    model.is_train = False
    f1_score = 0
    f1_score_2 = 0
    cd_val=0
    emd_val = 0
    for _,(image, points, surface_normals, __) in tenumerate(dataloader):
        image, points, surface_normals = image.to(device), points.to(device), surface_normals.to(device)
        predicted_mesh, _ = model(image, points, surface_normals)
        prediction = predicted_mesh.verts_padded()
        f1_score += f_score(prediction, points)
        f1_score_2 += f_score(prediction, points, 2e-4)
        chamferloss, _ = chamfer_distance(prediction, points)
        cd_val+=chamferloss
        emd_val+=emd(prediction, points)
    return f1_score/(len(dataloader)), f1_score_2/(len(dataloader)), cd_val, emd_val

def train(dataloader, model, optimizer):
    model.is_train= True
    model.train()

    size = len(dataloader.dataset)
    loss_temp = 0.0

    for batch, (image, points, surface_normals, dat_path) in tenumerate(dataloader):
        image, points, surface_normals = image.to(device), points.to(device), surface_normals.to(device)

        # Forward
        predicted_mesh, loss = model(image, points, surface_normals)

        # Back-propagation
        loss.backward()

        # Update parameters
        optimizer.step()
        optimizer.zero_grad()

        loss_temp += loss.item()

        if batch % 500 == 499:
            loss_temp /= 500.0
            current = batch * len(image)
            print(f"loss: {loss_temp:>7f}  [{current:>5d}/{size:>5d}]")
            loss_temp = 0.0

def train_loop(dataloader, model, optimizer, epoch_start, epoch_end, checkpoint_filename):
    for i in range(epoch_start, epoch_end):
        print(f"Epoch {i+1}\n-------------------------------")
        train(train_dataloader, model, optimizer)
        torch.save(model.state_dict(), f'checkpoints/{checkpoint_filename}.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run training script.')

    parser.add_argument("--data-base", help="Path to base directory of dataset")
    parser.add_argument("--train-data", help="Path to meta file for train dataset")
    parser.add_argument("--test-data", help="Path to meta file for test dataset")
    parser.add_argument("--skip-train", help="If true, don't train and jump right into testing phase", action="store_true", default=False)
    parser.add_argument("--checkpoint", help="Path to checkpoint to load")

    args = parser.parse_args()

    data_base_path = args.data_base
    meta_file_path = args.train_data
    meta_file_path_test = args.test_data
    checkpoint_path = args.checkpoint
    skip_train = args.skip_train

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet normalization
    ])
    train_data = ShapeNet(meta_file_path, data_base_path, transform)
    train_dataloader = DataLoader(train_data, batch_size=1, pin_memory=True)

    validation_data = ShapeNet(meta_file_path_test, data_base_path, transform)
    validation_dataloader = DataLoader(validation_data, batch_size=1, pin_memory=True)

    model = P2M(ellipsoid_path, camera_c, camera_f).to(device)

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))
        print(f'Checkpoint loaded from {checkpoint_path}')

    if not skip_train:
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, weight_decay=1e-5)
        time = datetime.datetime.now()
        checkpoint_filename = time.strftime('%m-%d_%H:%M')
        train_loop(train_dataloader, model, optimizer, 0, 40, checkpoint_filename)

        for parameter_group in optimizer.param_groups:
            parameter_group['lr'] = 1e-5

        train_loop(train_dataloader, model, optimizer, 40, 50, checkpoint_filename)

    f_score_value, f_score_value_2, cd_value, emd_value = test(validation_dataloader, model)
    print(f"f-score (tau): {f_score_value}", f"f-score (2-tau): {f_score_value_2}", f"chamfer distance: {cd_value}", f"emd: {emd_value}")
    
    print("Done!")
    