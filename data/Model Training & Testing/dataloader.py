import os
import torch
import pandas as pd

def load_dataset(root):
    csv = []
    csv_path = os.path.join(root, 'B')
    csv_files = os.listdir(csv_path)
    for csv_file in csv_files:
        csv_name = os.path.splitext(csv_file)[0] + '.xlsx'
        csv.append(os.path.join(csv_path, csv_name))
    return csv


def read_csv(path):
    data = pd.read_excel(path)
    data = data.iloc[:, 0].tolist()
    return data

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.csv = load_dataset(root)

    def __getitem__(self, index):
        csv = read_csv(self.csv[index])
        csv = torch.tensor(csv)
        csv = csv.unsqueeze(0)
        return csv

    def __len__(self):
        return len(self.csv)

def get_loader(root, batch_size, num_works=0):
    dataset = Dataset(root)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle = True)
    return dataloader