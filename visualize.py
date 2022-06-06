import torch
from torch.utils.data import DataLoader
from pytorch3d.vis.plotly_vis import plot_batch_individually
from torchvision import transforms
from tqdm.contrib import tenumerate

from dataset import ShapeNet
from model import P2M
from metrics import f_score

@torch.no_grad()
def visualize(dataloader, model):
    model.eval()
    model.is_train = False
    f1_max = 0.0
    f1_temp = 0.0
    plot_filename = "Untitled.png"
    fig = None
    for _,(image, points, surface_normals, dat_path) in tenumerate(dataloader):
        image, points, surface_normals = image.to(device), points.to(device), surface_normals.to(device)
        predicted_mesh, _ = model(image, points, surface_normals)

        prediction = predicted_mesh.verts_padded()
        f1_temp = f_score(prediction, points)

        if f1_temp > f1_max:
            f1_max = f1_temp
            fig = plot_batch_individually(predicted_mesh)
            plot_filename = f'{dat_path[0].split("/")[-3]}-{dat_path[0].split("/")[-1]}.png'

    if fig:
        fig.write_image(plot_filename)

ellipsoid_path = "./data/initial_ellipsoid.ply"
camera_c = [112.0, 112.0]
camera_f = [250.0, 250.0]

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    data_base_path = "/root/Pixel2Mesh-reference/datasets/data/shapenet/data_tf"
    meta_file_path_test = "./data/meta/category/test_list_car.txt"
    checkpoint_path = "./checkpoints/full-model-40-epochs.pth" 

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet normalization
    ])
    
    validation_data = ShapeNet(meta_file_path_test, data_base_path, transform)
    validation_dataloader = DataLoader(validation_data, batch_size=1, pin_memory=True)
    model = P2M(ellipsoid_path, camera_c, camera_f).to(device)

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))
        print(f'Checkpoint loaded from {checkpoint_path}')

    visualize(validation_dataloader, model)

    print("Done!")