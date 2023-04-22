
# import torch
# import torch.nn as nn
# import torchvision.models as models

# # 定义ResNet18模型
# class ResNet18(nn.Module):
#     def __init__(self, num_classes=1000):
#         super(ResNet18, self).__init__()
#         self.model = models.resnet18(pretrained=True)
#         num_ftrs = self.model.fc.in_features
#         self.model.fc = nn.Linear(num_ftrs, num_classes)

#     def forward(self, x):
#         x = self.model(x)
#         return x

# # 创建模型实例
# model = ResNet18()

# # 导出ONNX模型
# input_tensor = torch.randn(1, 3, 224, 224)  # 创建一个示例输入张量
# torch.onnx.export(model, input_tensor, "resnet18.onnx", verbose=True)


import torch
import torchvision

# 加载训练好的模型
model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 5)
model.load_state_dict(torch.load('/home/ruoji/MNN/data/resnet18_flowers.pth'))

# 设置输入张量的形状
batch_size = 1
input = torch.randn(batch_size, 3, 224, 224)


input_name = 'input'
output_name = 'output'
# 将模型转换为ONNX格式
torch.onnx.export(model, input, 
                  'resnet18_flower_onnx.onnx', 
                  input_names = [input_name],
                  output_names = [output_name],
                  verbose=True,
                  opset_version=11,
                  dynamic_axes={input_name: {0: 'batch_size'},
                                output_name: {0: 'batch_size'}})


