import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torchvision import datasets
from torch.autograd import Variable
from torchvision.models.resnet import resnet18

import json
import os
# 定义训练过程

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    sum_loss = 0
    total_num = len(train_loader.dataset)
    print(total_num, len(train_loader))
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).to(device), Variable(target).to(device)
        fea,output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print_loss = loss.data.item()
        sum_loss += print_loss
        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item()))
    ave_loss = sum_loss / len(train_loader)
    print('epoch:{},loss:{}'.format(epoch, ave_loss))

Best_ACC=0
# 验证过程
@torch.no_grad()
def val(model, device, test_loader):
    global Best_ACC
    model.eval()
    test_loss = 0
    correct = 0
    total_num = len(test_loader.dataset)
    print(total_num, len(test_loader))
    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data).to(device), Variable(target).to(device)
            fea,output = model(data)
            loss = criterion(output, target)
            _, pred = torch.max(output.data, 1)
            correct += torch.sum(pred == target)
            print_loss = loss.data.item()
            test_loss += print_loss
        correct = correct.data.item()
        acc = correct / total_num
        avgloss = test_loss / len(test_loader)
        if acc > Best_ACC:
            torch.save(model, file_dir + '/' + 'best.pth')
            Best_ACC = acc
        print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            avgloss, correct, len(test_loader.dataset), 100 * acc))
        return acc



if __name__ == '__main__':
    # 创建保存模型的文件夹
    file_dir = 'resnet'
    if os.path.exists(file_dir):
        print('true')

        os.makedirs(file_dir, exist_ok=True)
    else:
        os.makedirs(file_dir)

    # 设置全局参数
    modellr = 1e-4
    BATCH_SIZE = 16
    EPOCHS = 100
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据预处理7
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 3.0)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.44127703, 0.4712498, 0.43714803], std=[0.18507297, 0.18050247, 0.16784933])

    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.44127703, 0.4712498, 0.43714803], std=[0.18507297, 0.18050247, 0.16784933])
    ])

    # 读取数据
    dataset_train = datasets.ImageFolder('data/train', transform=transform)
    dataset_test = datasets.ImageFolder("data/val", transform=transform_test)
    with open('class.txt', 'w') as file:
        file.write(str(dataset_train.class_to_idx))
    with open('class.json', 'w', encoding='utf-8') as file:
        file.write(json.dumps(dataset_train.class_to_idx))
    # 导入数据
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

    # 实例化模型并且移动到GPU
    criterion = nn.CrossEntropyLoss()

    model_ft = resnet18()
    print(model_ft)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 12)
    model_ft.to(DEVICE)
    # 选择简单暴力的Adam优化器，学习率调低
    optimizer = optim.Adam(model_ft.parameters(), lr=modellr)
    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-9)
    # 训练
    val_acc_list= {}
    for epoch in range(1, EPOCHS + 1):
        train(model_ft, DEVICE, train_loader, optimizer, epoch)
        cosine_schedule.step()
        acc=val(model_ft, DEVICE, test_loader)
        val_acc_list[epoch]=acc
        with open('result_student.json', 'w', encoding='utf-8') as file:
            file.write(json.dumps(val_acc_list))
    torch.save(model_ft, 'resnet/model_final.pth')


