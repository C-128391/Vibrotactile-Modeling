import os
import pandas as pd
import torch

def normalize_data(data):
    min_val = torch.min(data)
    max_val = torch.max(data)
    normalized_data = data / (max_val - min_val)
    return normalized_data

folder_path = r"file/to/path"
new_file_path = r"file/to/path"

for filename in os.listdir(folder_path):
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_excel(file_path)
        tensor_data = torch.tensor(df.values, dtype=torch.float)
        normalized_data = normalize_data(tensor_data)

        df_normalized = pd.DataFrame(normalized_data.numpy(), columns=df.columns)

        new_file_path = os.path.join(folder_path, filename)
        df_normalized.to_excel(new_file_path, index=False)
