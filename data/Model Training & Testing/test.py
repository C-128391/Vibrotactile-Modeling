import matplotlib.pyplot as plt
from encoder import *
import numpy as np
import os
import torch
import pandas as pd
from math import sqrt
import openpyxl

#Validate the results on the test set.

workbook = openpyxl.Workbook()
sheet = workbook.active

model2 = Generator().cuda()
model2.load_state_dict(torch.load('model/best_model.pt')) #The path corresponding to the trained model.
model2.eval()

def read_csv(path):
    data = pd.read_excel(path)
    data = data.iloc[:, 0].tolist()
    return data

def get_file_names(folder_path):
    file_names = []
    for file_name in os.listdir(folder_path):
        file_names.append(file_name)
    return file_names

def calculate_rmse(actual, predicted):
    n = len(actual)
    mse = np.mean((np.array(actual) - np.array(predicted)) ** 2)
    rmse = sqrt(mse)
    return rmse

def pix2pix(i):
    csv_name = file_names[i]
    csv_name = os.path.splitext(csv_name)[0] + '.xlsx'  # Get file name (without extension)
    end_index = csv_name.index('-')
    content = csv_name[:end_index]
    image_name = content + '_square.png'
    return csv_name, image_name


csv_path = "test/B"

file_names = get_file_names(csv_path)
print(len(file_names))

generate = []
real = []

for i in range(100):
    csv_name = file_names[i+463*9]
    c_path = os.path.join(csv_path, csv_name)
    csv = torch.tensor(read_csv(c_path)).unsqueeze(0).to(device)
    x_t, x_pred_t = model2(csv)
    x_t = x_t.flatten().tolist()
    x_pred_t = x_pred_t.flatten().tolist()
    generate += x_pred_t
    real += x_t

time_signal1 = real
time_signal2 = generate
plt.plot(time_signal1)
plt.plot(time_signal2)
plt.show()






