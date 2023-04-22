import torchvision.models as models
import torch.nn as nn
import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from mnncompress.pytorch import LSQQuantizer
Quantizer = LSQQuantizer


# batch_size = 16
# train_loader = torch.utils.data.DataLoader(
#     datasets.ImageFolder('/home/ruoji/MNN/data/flowers/train', transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])),
#     batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

# test_loader = torch.utils.data.DataLoader(
#     datasets.ImageFolder('/home/ruoji/MNN/data/flowers/val', transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])),
#     batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

# 定义数据的路径和预处理方法
data_dir = '/home/ruoji/MNN/data/flowers'
train_dir = data_dir + '/train'
val_dir = data_dir + '/val'

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# 加载数据
train_data = datasets.ImageFolder(train_dir, data_transforms['train'])
val_data = datasets.ImageFolder(val_dir, data_transforms['val'])
train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=4)


import torch.nn.functional as F
criterion = nn.CrossEntropyLoss()

def train(epoch):
    quant_model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = quant_model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# def test():
#     quant_model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = quant_model(data)
#             test_loss += F.cross_entropy(output, target, reduction='sum').item()
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)
#     accuracy = 100. * correct / len(test_loader.dataset)
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset), accuracy))


def train2(quant_model):
    quant_model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = quant_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    scheduler.step()

    quant_model.eval()
    num_correct = 0
    num_total = 0
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        num_correct += (predicted == labels).sum().item()
        num_total += labels.size(0)

    epoch_loss = running_loss / len(train_data)
    epoch_acc = num_correct / num_total
    print('Epoch {} - Loss: {:.4f} Acc: {:.4f}'.format(epoch+1, epoch_loss, epoch_acc))

# 你的模型代码
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)  # 将全连接层改为5分类

# 加载已经训练好的模型，可以是剪枝之后的
model.load_state_dict(torch.load("/home/ruoji/MNN/data/resnet18_flowers_best.pth"))

# 将模型进行转换，并使用转换后的模型进行训练，测试
# retain_sparsity=True表示待量化的float模型是稀疏模型，希望叠加量化训练
# 更多配置请看API部分
# 注意在model还在cpu上，任何分布式还未生效前调用
quantizer = Quantizer(model, retain_sparsity=False)
quant_model = quantizer.convert()

# 单机或分布式，根据你已有的代码来
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
quant_model.to(device)

optimizer = optim.SGD(quant_model.parameters(), lr=0.001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

quantizer.resume_qat_graph()

for epoch in range(1, 20 + 1):
    train2(quant_model)
    # test()

quantizer.strip_qat_ops()
# 保存模型，注意index，即模型和保存MNN模型压缩参数文件是一一对应的
quant_model.eval()
torch.save(quant_model.state_dict(), "quant_model_20ep.pt")
x = torch.randn(1, 3, 224, 224).to(device)
torch.onnx.export(quant_model, x, "quant_model_20ep.onnx")
# 保存MNN模型压缩参数文件，如果进行量化的模型有剪枝，
# 请将剪枝时生成的MNN模型压缩参数文件 "compress_params.bin" 文件在下方传入，并将 append 设置为True
quantizer.save_compress_params("quant_model_20ep.onnx", "compress_params_20ep.bin", append=False)