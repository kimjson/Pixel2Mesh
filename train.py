import datetime

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchinfo import summary
from tqdm.contrib import tenumerate

from metrics import f_score
from dataset import ShapeNet
from model import P2M

device = "cuda" if torch.cuda.is_available() else "cpu"

# TODO: replace with command line parameters
meta_file_path = "/root/Pixel2Mesh-reference/datasets/data/shapenet/meta/train_tf.txt"
meta_file_path_test = "/root/Pixel2Mesh-reference/datasets/data/shapenet/meta/test_tf.txt"

data_base_path = "/root/Pixel2Mesh-reference/datasets/data/shapenet/data_tf"
ellipsoid_path = "/root/Pixel2Mesh/data/initial_ellipsoid.ply"
camera_c = [112.0, 112.0]
camera_f = [250.0, 250.0]

@torch.no_grad()
def test(dataloader, model):
    model.eval()
    model.is_train = False
    f1_score = 0
    for _,(image, points, surface_normals) in tenumerate(dataloader):
        image, points, surface_normals = image.to(device), points.to(device), surface_normals.to(device)
        predicted_mesh, _ = model(image, points, surface_normals)
        prediction = predicted_mesh.verts_padded()
        f1_score += f_score(prediction,points)
    return f1_score/(len(dataloader))

def train(dataloader, model, optimizer):
    model.is_train= True
    model.train()

    size = len(dataloader.dataset)
    loss_temp = 0.0

    for batch, (image, points, surface_normals) in tenumerate(dataloader):
        image, points, surface_normals = image.to(device), points.to(device), surface_normals.to(device)

        predicted_mesh, loss = model(image, points, surface_normals)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_temp += loss.item()

        if batch % 500 == 499:
            loss_temp /= 500.0
            current = batch * len(image)
            print(f"loss: {loss_temp:>7f}  [{current:>5d}/{size:>5d}]")
            loss_temp = 0.0

def train_loop(dataloader, model, optimizer, epoch_start, epoch_end, checkpoint_filename):
    f_score_best_value = 0
    for i in range(epoch_start, epoch_end):
        print(f"Epoch {i+1}\n-------------------------------")
        train(train_dataloader, model, optimizer)
        f_score_value = test(validation_dataloader, model)
        if f_score_value > f_score_best_value :
            f_score_best_value = f_score_value
            torch.save(model.state_dict(), f'checkpoints/{checkpoint_filename}.pth')
        print(f"f-score: {f_score_value}")

    print(f"best f-score: {f_score_best_value}")

if __name__ == "__main__":
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

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, weight_decay=1e-5)
    time = datetime.datetime.now()
    checkpoint_filename = time.strftime('%m-%d_%H:%M')
    train_loop(train_dataloader, model, optimizer, 0, 40, checkpoint_filename)

    for parameter_group in optimizer.param_groups:
        parameter_group['lr'] = 1e-5

    train_loop(train_dataloader, model, optimizer, 40, 50, checkpoint_filename)
    
    print("Done!")
    