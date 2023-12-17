import pandas as pd
import os
import openpyxl
import torchvision.transforms as transforms
from PIL import Image
from encoder import make_model

workbook = openpyxl.Workbook()
sheet = workbook.active

ResNet = make_model()

folder_path1 = r'file/to/path'
file_names1 = os.listdir(folder_path1)

folder_path2 = r'file/to/path'
file_names2 = os.listdir(folder_path2)

folder_path3 = r'file/to/path'
file_names3 = os.listdir(folder_path3)

images_path ='file/to/path'

images = []

for csv_file in file_names3:
    csv_name = os.path.splitext(csv_file)[0] + '.xlsx'
    end_index = csv_name.index('.')
    start_index = csv_name.index('_')
    content = csv_name[start_index+1:end_index]
    image_name = content + '_square.png'
    images.append(os.path.join(images_path, image_name))

def rgb_to_gray(image):
    gray_image = image.convert('L')
    gray_image_rgb = gray_image.convert('RGB')
    return gray_image_rgb

for i in range(10):
    df1 = pd.read_excel(file_names1[i])
    df2 = pd.read_excel(file_names2[i])
    df3 = pd.read_excel(file_names3[i])
    image = transforms.ToTensor()(rgb_to_gray(Image.open(images[i]))).unsqueeze(0)

    column_values1 = df1.iloc[:, 0].tolist()[:39000]
    column_values2 = df2.iloc[:, 0].tolist()[:39000]
    column_values3 = df3.iloc[:, 0].tolist()[:39000]

    image_feature = ResNet(image).flatten().tolist()
    steps = 20
    lens = 220
    epoch = int((len(column_values1) - lens) / steps)
    epoch = epoch - 10
    print(epoch)

    for o in range(epoch):
        list1 = []
        list1 += image_feature
        for j in column_values1[200 + 20 * o:20 * o + 220]:
            list1.append(j)
        for k in column_values2[200 + 20 * o:20 * o + 220]:
            list1.append(k)
        for l in column_values3[20 * o:20 * o + 220]:
            list1.append(l)
        print(len(list1))

        df = pd.DataFrame(list1, columns=None)

        start_index = file_names1[i+90].index('_') + 1
        end_index = file_names1[i+90].index('.')
        content = file_names1[i+90][start_index:end_index]

        df = pd.DataFrame({'data': list1})
        excel_file = r"file/to/path" % content + "-" + "%s" % o + ".xlsx"
        df.to_excel(excel_file, index=False)



