import os
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from PIL import Image
from encoder import *
import csv
from googletrans import Translator

# Used to obtain t-SNE images.






vgg16 = models.vgg16(pretrained=True)
vgg16.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

model2 = Generator().cuda()
model2.load_state_dict(torch.load('model/best_model.pt')) #The path corresponding to the trained model.
model2.eval()

data = model2.linear1
data = data.to('cpu')

image_folder = r"E:\pycharm projeces\vibrotactile display\original data\data_deal\image" #The folder path corresponding to the texture images.
image_files = os.listdir(image_folder)

features = []
image_paths = []

for file_name in image_files:
    image_path = os.path.join(image_folder, file_name)
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        feature = vgg16(image_tensor)
        feature = data(feature)

    features.append(feature.squeeze().numpy())
    image_paths.append(image_path)

features = torch.tensor(features)

X_tsne = TSNE(n_components=2, random_state=64).fit_transform(features)

plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])

for i, path in enumerate(image_paths):
    translator = Translator()
    english_label = os.path.basename(path)[:-11]
    chinese_label = translator.translate(english_label, dest='zh-CN').text

    plt.annotate(chinese_label, (X_tsne[i, 0], X_tsne[i, 1]), fontsize=4)
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("t-SNE Visualization of Image Features")

plt.savefig('t-sne.png', dpi = 300)
plt.show()

data = [(os.path.basename(path), coord[0], coord[1]) for path, coord in zip(image_paths, X_tsne)]


output_file = 'tsne_coordinates.csv'

with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Filename', 'X', 'Y'])
    writer.writerows(data)

for i, path in enumerate(image_paths):
    plt.annotate(os.path.basename(path), (X_tsne[i, 0], X_tsne[i, 1]), fontsize=4)

