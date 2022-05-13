import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import ShapeNet
from model import P2M

device = "cuda" if torch.cuda.is_available() else "cpu"

# TODO: replace with command line parameters
meta_file_path = "/root/Pixel2Mesh-reference/datasets/data/shapenet/meta/train_tf.txt"
data_base_path = "/root/Pixel2Mesh-reference/datasets/data/shapenet/data_tf"
ellipsoid_path = "/root/Pixel2Mesh/data/initial_ellipsoid.ply"
camera_c = [112.0, 112.0]
camera_f = [250.0, 250.0]

def loss_function(predicted_mesh, true_points, true_normals):
    return 0

def train(dataloader, model, loss_function, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (image, points, surface_normals) in enumerate(dataloader):
        image, points, surface_normals = image.to(device), points.to(device), surface_normals.to(device)

        # Compute prediction error
        predicted_mesh = model(image, dataloader.dataset.camera_c, dataloader.dataset.camera_f)
        loss = loss_function(predicted_mesh, points, surface_normals)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(image)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet normalization
    ])
    train_data = ShapeNet(meta_file_path, data_base_path, camera_c, camera_f, transform)
    train_dataloader = DataLoader(train_data, batch_size=1)

    model = P2M(ellipsoid_path).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    epochs = 2
    
    for i in range(epochs):
        print(f"Epoch {i+1}\n-------------------------------")
        train(train_dataloader, model, loss_function, optimizer)

    print("Done!")
    