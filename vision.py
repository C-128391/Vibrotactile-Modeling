import os
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from PIL import Image
from encoder import *
import csv

transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
])

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class ModifiedResidualStudentModel(nn.Module):
    def __init__(self, num_classes=64, channels=16):
        super(ModifiedResidualStudentModel, self).__init__()
        self.conv1 = nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual_block = ResidualBlock(channels, channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.residual_block(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


model2 = ModifiedResidualStudentModel()
model2.load_state_dict(torch.load('file/to/path'))
model2.eval()

data = model2

data = data.to('cpu')

image_folder = r"file/to/path"
image_files = os.listdir(image_folder)

features = []
image_paths = []

for file_name in image_files:
    image_path = os.path.join(image_folder, file_name)
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        feature = data(image_tensor)

    features.append(feature.squeeze().numpy())
    image_paths.append(image_path)

features = torch.tensor(features)

X_tsne = TSNE(n_components=2, random_state=64).fit_transform(features)

plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
for i, path in enumerate(image_paths):
    plt.annotate(os.path.basename(path)[:-11], (X_tsne[i, 0], X_tsne[i, 1]), fontsize=4, fontproperties='SimHei')
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("t-SNE Visualization of Image Features")

plt.savefig('file/to/path', dpi = 300)
plt.show()

data = [(os.path.basename(path), coord[0], coord[1]) for path, coord in zip(image_paths, X_tsne)]

output_file = 'file/to/path'

with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Filename', 'X', 'Y'])
    writer.writerows(data)

for i, path in enumerate(image_paths):
    plt.annotate(os.path.basename(path), (X_tsne[i, 0], X_tsne[i, 1]), fontsize=4)

