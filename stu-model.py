import os
import torch.optim as optim
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset
from encoder import *
import openpyxl

workbook = openpyxl.Workbook()
sheet = workbook.active

def make_model():
    model = models.vgg16(pretrained=True)
    model = model.eval()
    return model

generate = Generator().cuda()
generate.load_state_dict(torch.load('file/to/path'))
generate.eval()


class Teacher(nn.Module):
    def __init__(self):
        super(Teacher, self).__init__()
        self.vgg = make_model()
        self.linear = generate.linear1

    def forward(self, x):
        x = self.vgg(x)
        x = self.linear(x)
        return x

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

device = torch.device("cuda:0")
teacher_model = Teacher().to(device)
student_model = ModifiedResidualStudentModel().to(device)

for param in teacher_model.parameters():
    param.requires_grad = False

class FeatureExtractionDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img,0

data_dir = r'file/to/path'


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


dataset = FeatureExtractionDataset(data_dir, transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
optimizer = optim.Adam(student_model.parameters(), lr=0.001)

num_epochs = 3000
train_loss_list = []
for epoch in range(num_epochs):
    trn = []
    for inputs, _ in dataloader:
        student_outputs = student_model(inputs.to(device))
        target_features = teacher_model(inputs.to(device))

        loss = nn.functional.l1_loss(student_outputs, target_features)
        trn.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = (sum(trn) / len(dataloader))
    train_loss_list.append(train_loss)

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss}')
    torch.save(student_model.state_dict(), 'stu_model.pt')

for j in range(len(train_loss_list)):
    sheet.cell(row=j + 1, column=1, value=train_loss_list[j])
workbook.save(r"知识蒸馏_loss.xlsx")

plt.plot(train_loss_list)
plt.show()

