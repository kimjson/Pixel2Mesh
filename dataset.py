import os, pickle

from torch.utils.data import Dataset
from torchvision.io import read_image

class ShapeNet(Dataset):
    def __init__(self, meta_file_path, data_base_path, camera_c, camera_f, transform=None):
        # read list file
        with open(meta_file_path, 'r') as meta_file:
            self.dat_file_paths = meta_file.readlines()
            self.png_file_paths = dat_file_paths[:]

            for i, path in enumerate(self.dat_file_paths):
                path_without_prefix = path.split("Data/ShapeNetP2M/")[1]
                self.dat_file_paths[i] = os.path.join(data_base_path, path_without_prefix)
                self.png_file_paths[i] = dat_file_paths[i].replace(".dat", ".png")

        self.camera_c = camera_c
        self.camera_f = camera_f
        self.transform = transform
                

    def __len__(self):
        return len(self.dat_file_paths)

    def __get_item__(self, index):
        image_path = self.png_file_paths[index]
        image = read_image(image_path)
        if self.transform:
            image = self.transform(image)

        dat_path = self.dat_file_paths[index]
        with open(dat_path, "rb") as dat_file:
            dat = pickle.load(dat_file, encoding="latin1")
            points = dat[:, :3]
            surface_normals = dat[:, 3:]

        return image, points, surface_normals, self.camera_c, self.camera_f