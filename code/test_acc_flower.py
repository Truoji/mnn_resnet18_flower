import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import os
import shutil
import argparse
import numpy as np

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models

import onnxruntime as ort

import MNN

from bisect import bisect_right
import time
import math


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='/home/ruoji/MNN/data/flowers', type=str, help='trainset directory')
parser.add_argument('--dataset', default='ImageNet', type=str, help='Dataset name')
parser.add_argument('--arch', default='resnet50', type=str, help='network architecture')
parser.add_argument('--batch-size', type=int, default=16, help='batch size')
parser.add_argument('--num-workers', type=int, default=8, help='number workers')
# parser.add_argument('--model-path', default='/home/ruoji/MNN/data/resnet18_flowers_best.pth', type=str, help='pretrained weights')
# parser.add_argument('--model-path', default='/home/ruoji/MNN/data/resnet18_flower_onnx.onnx', type=str, help='pretrained weights')
# parser.add_argument('--model-path', default='/home/ruoji/MNN/data/resnet18_flower_fp32.mnn', type=str, help='pretrained weights')
# parser.add_argument('--model-path', default='/home/ruoji/MNN/data/resnet18_flower_int8.mnn', type=str, help='pretrained weights')
# parser.add_argument('--model-path', default='/home/ruoji/MNN/quant_model_20ep.onnx', type=str, help='pretrained weights')
parser.add_argument('--model-path', default='/home/ruoji/MNN/renet18_quant_20ep.mnn', type=str, help='pretrained weights')


parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--manual_seed', type=int, default=0)
# parser.add_argument('--model', type=str, default='torch', help='torch, onnx, mnn')
# parser.add_argument('--model', type=str, default='onnx', help='torch, onnx, mnn')
parser.add_argument('--model', type=str, default='mnn', help='torch, onnx, mnn')


# global hyperparameter set
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

num_classes = 5

test_set = datasets.ImageFolder(
    os.path.join(args.data, 'val'),
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
]))

# testloader = torch.utils.data.DataLoader(
#     test_set, batch_size=args.batch_size, shuffle=False,
#     num_workers=args.num_workers, pin_memory=True)

testloader = torch.utils.data.DataLoader(
    test_set, batch_size=1, shuffle=False,
    num_workers=args.num_workers, pin_memory=True)
# --------------------------------------------------------------------------------------------

# Model
print('==> Building model..')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.model == 'torch':
    # net = models.resnet18(pretrained=True).cuda()
    model = models.resnet18(pretrained=True).cuda()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5).cuda()  # 将全连接层改为5分类
    model.load_state_dict(torch.load("/home/ruoji/MNN/data/resnet18_flowers_best.pth"))
    cudnn.benchmark = True  

# Load onnx Model
if args.model == 'onnx':
    session = ort.InferenceSession(args.model_path, providers=['CUDAExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

# Load mnn Model
if args.model == 'mnn':
    interpreter = MNN.Interpreter(args.model_path)
    interpreter.setCacheFile('.tempcache')
    config = {}
    config['precision'] = 'low'
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)


def correct_num(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:, :k].float().sum()
        res.append(correct_k.item())
    return res



correct1 = 0
correct5 = 0
total = 0
sum_time = time.time()
with torch.no_grad():
    batch_start_time = time.time()
    for batch_idx, (inputs, target) in enumerate(testloader):
        if args.model == 'torch':
            model.eval()
            inputs, target = inputs.cuda(), target.cuda()
            logits = model(inputs)
        elif args.model == 'onnx':
            inputs = {input_name: to_numpy(inputs)}
            logits = session.run([output_name], inputs)[0]
            logits = torch.from_numpy(logits)
        elif args.model == 'mnn':
            interpreter.resizeTensor(input_tensor, inputs.shape)
            interpreter.resizeSession(session)
            tmp_input = MNN.Tensor(inputs.shape, MNN.Halide_Type_Float,\
                    to_numpy(inputs), MNN.Tensor_DimensionType_Caffe)
            input_tensor.copyFrom(tmp_input)
            interpreter.runSession(session)
            logits = interpreter.getSessionOutput(session)
            out_shape = logits.getShape()
            #constuct a tmp tensor and copy/convert in case output_tensor is nc4hw4
            tmp_output = MNN.Tensor(out_shape, MNN.Halide_Type_Float, np.ones(out_shape).astype(np.float32), MNN.Tensor_DimensionType_Caffe)
            logits.copyToHostTensor(tmp_output)
            out_data = tmp_output.getNumpyData()
            logits = out_data.reshape(out_shape)
            logits = torch.from_numpy(logits)

        print('batch_idx:{}/{}, Duration:{:.2f}'.format(batch_idx, len(testloader), time.time()-batch_start_time))
        batch_start_time = time.time()

        prec1, prec5 = correct_num(logits, target, topk=(1, 5))
        correct1 += prec1
        correct5 += prec5
        total += target.size(0)

    acc1 = round(correct1/total, 4)
    acc5 = round(correct5/total, 4)
    
    print('Test accuracy_1:{:.4f}\n'
                'Test accuracy_5:{:.4f}\n'
                .format(acc1, acc5))

print('avg times:', (time.time()-sum_time)/50000)
