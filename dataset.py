import os, pickle

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode

class ShapeNet(Dataset):
    def __init__(self, meta_file_path, data_base_path, transform=None):
        # read list file
        with open(meta_file_path, 'r') as meta_file:
            # TODO: Sample uniformly across category - e.g. 1k models per category
            self.dat_file_paths = meta_file.readlines()
            self.png_file_paths = self.dat_file_paths[:]

            for i, path in enumerate(self.dat_file_paths):
                path_without_prefix = path.strip().split("Data/ShapeNetP2M/")[1]
                self.dat_file_paths[i] = os.path.join(data_base_path, path_without_prefix)
                self.png_file_paths[i] = self.dat_file_paths[i].replace(".dat", ".png")

        self.transform = transform
                

    def __len__(self):
        return len(self.dat_file_paths)

    def __getitem__(self, index):
        image_path = self.png_file_paths[index]
        image = read_image(image_path, ImageReadMode.RGB)
        if self.transform:
            image = self.transform(image)

        dat_path = self.dat_file_paths[index]
        with open(dat_path, "rb") as dat_file:
            dat = pickle.load(dat_file, encoding="latin1")
            points = torch.tensor(dat[:, :3])
            surface_normals = torch.tensor(dat[:, 3:])

        return image, points, surface_normals, dat_path