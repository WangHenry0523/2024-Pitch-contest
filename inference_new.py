import torch.nn as nn
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
class ResidualBlock(nn.Module):
    # 初始化函數
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        # 調用父類的初始化函數
        super(ResidualBlock, self).__init__()
        # 第一個卷積層，包括卷積、批量正規化和ReLU激活
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        # 第二個卷積層，包括卷積和批量正規化
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        # 下採樣層，用於調整殘差的維度
        self.downsample = downsample
        # ReLU激活函數
        self.relu = nn.ReLU()
        # 輸出通道數
        self.out_channels = out_channels
    # 前向傳播函數
    def forward(self, x):
        # 保存輸入作為殘差
        residual = x
        # 通過第一個卷積層
        out = self.conv1(x)
        out = self.conv2(out)# 通過第二個卷積層
        # 如果有下採樣，對殘差進行下採樣
        if self.downsample:
            residual = self.downsample(x)
        # 將輸出和殘差相加
        out += residual
        # 通過ReLU激活函數
        out = self.relu(out)
        return out
# 定義ResNet類，繼承自nn.Module
class ResNet(nn.Module):
    # 初始化函數
    def __init__(self, block, layers, num_classes = 2):
        # 調用父類的初始化函數
        super(ResNet, self).__init__()
        # 初始化輸入通道數
        self.inplanes = 64
        # 第一個卷積層，包括卷積、批量正規化和ReLU激活
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        # 最大池化層
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        # 創建四個殘差層
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        # 平均池化層
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # 全連接層
        self.fc = nn.Linear(512, num_classes)
    # 創建殘差層的函數
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # 判斷是否需要下採樣
        if stride != 1 or self.inplanes != planes:

            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        # 初始化殘差塊列表
        layers = []
        # 添加第一個殘差塊，可能包含下採樣
        layers.append(block(self.inplanes, planes, stride, downsample))
        # 更新輸入通道數
        self.inplanes = planes
        # 添加其他殘差塊
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        # 返回殘差層
        return nn.Sequential(*layers)

    # 前向傳播函數
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# classes_3 = ["n","m","k"]
# model_3 = torchvision.models.resnet18()
# num_ftrs = model_3.fc.in_features
# model_3.fc = nn.Linear(num_ftrs, len(classes_3))
# checkpoint3 = torch.load('best_single_isic_model_resnet18 (1)_mk.pth', map_location=torch.device('cpu'))
# model_3.load_state_dict(checkpoint3)
# model_3.eval()

# classes_2 = ["n","k"]
# model_2 = torchvision.models.resnet18()
# num_ftrs2 = model_2.fc.in_features
# model_2.fc = nn.Linear(num_ftrs2, len(classes_2))
# checkpoint2 = torch.load('best_single_isic_model_resnet18V2 (1)_nk.pth', map_location=torch.device('cpu'))
# model_2.load_state_dict(checkpoint2)
# model_2.eval()

# # 载入模型mn
# model = ResNet(ResidualBlock, [2,2,2,2])
# # 三个类别
# classes = ["m", "n"]
# checkpoint = torch.load('best_model_resnet18_nm.pth', map_location=torch.device('cpu'))
# model.load_state_dict(checkpoint)
# model.eval()

# 图像预处理步骤
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.RandomRotation(degrees=15),  # 随机旋转图像，范围在 -15 到 15 度之间
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 推理函数
def predict(image_path):
    # 载入模型mn
    model = ResNet(ResidualBlock, [2, 2, 2, 2])
    # 三个类别
    classes = ["melanoma", "nevus"]
    checkpoint = torch.load('best_modelresnet18_success_1.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()
    probability_m_n = []
    # 处理图片
    input_image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(input_image)
    input_tensor = input_tensor.unsqueeze(0)
    # 推理
    with torch.no_grad():
        output = model(input_tensor)
        output_max = output.argmax(1)
        probability_m_n = torch.nn.functional.softmax(output[0], dim=0) * 100
        # print(probabilities,format(classes[output_max.item()]))
        m_n = format(classes[output_max.item()])
    return m_n, probability_m_n

def predict3(image_path):
    classes_3 = ["nevus", "melanoma", 'seborrheic_keratosis']
    model_3 = torchvision.models.resnet18()
    num_ftrs = model_3.fc.in_features
    model_3.fc = nn.Linear(num_ftrs, len(classes_3))
    checkpoint3 = torch.load('best_single_isic_model_resnet18 (1)_mk.pth', map_location=torch.device('cpu'))
    model_3.load_state_dict(checkpoint3)
    model_3.eval()
    probabilities_m_k = []
    # 处理图片
    input_image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(input_image)
    input_tensor = input_tensor.unsqueeze(0)
    # 推理
    with torch.no_grad():
        output = model_3(input_tensor)
        output_max = output.argmax(1)
        probabilities_m_k = torch.nn.functional.softmax(output[0], dim=0) * 100
        # print(probabilities,format(classes_3[output_max.item()]))
        m_k = format(classes_3[output_max.item()])
    return m_k,probabilities_m_k

def predict2(image_path):
    classes_2 = ["nevus", "seborrheic_keratosis"]
    model_2 = torchvision.models.resnet18()
    num_ftrs2 = model_2.fc.in_features
    model_2.fc = nn.Linear(num_ftrs2, len(classes_2))
    checkpoint2 = torch.load('best_single_isic_model_resnet18V2 (1)_nk.pth', map_location=torch.device('cpu'))
    model_2.load_state_dict(checkpoint2)
    model_2.eval()
    probabilities_n_k = []
    # 处理图片
    input_image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(input_image)
    input_tensor = input_tensor.unsqueeze(0)
    # 推理
    with torch.no_grad():
        output = model_2(input_tensor)
        output_max = output.argmax(1)
        probabilities_n_k = torch.nn.functional.softmax(output[0], dim=0) * 100
        # print(probabilities,format(classes_2[output_max.item()]))
        n_k = format(classes_2[output_max.item()])
    return  n_k, probabilities_n_k

# 预测指定文件夹中的所有图像
def predict_images(image_path, save_path1, save_path2, save_path3):
    classes = ["melanoma", "nevus"]
    classes_3 = ["nevus", "melanoma", 'seborrheic_keratosis']
    classes_2 = ["nevus", "seborrheic_keratosis"]
    m_n, probabilities_m_n = predict(image_path)
    n_k, probabilities_n_k = predict2(image_path)
    m_k, probabilities_m_k = predict3(image_path)
    probabilitiesssss_m_n = []
    probabilitiesssss_n_k = []
    probabilitiesssss_m_k = []
    for i in range(2):
        probabilitiesssss_m_n.append(probabilities_m_n[i].item())
    plt.figure()
    plt.bar(classes[:], probabilitiesssss_m_n[:])
    plt.xlabel('Classes')
    plt.ylabel('Probability')
    plt.title('Class Probabilities')
    plt.savefig(save_path1)
    for i in range(2):
        probabilitiesssss_n_k.append(probabilities_n_k[i].item())
    plt.figure()
    plt.bar(classes_2[:], probabilitiesssss_n_k[:])
    plt.xlabel('Classes')
    plt.ylabel('Probability')
    plt.title('Class Probabilities')
    plt.savefig(save_path2)
    for i in range(3):
        probabilitiesssss_m_k.append(probabilities_m_k[i].item())
    plt.figure()
    plt.bar(classes_3[1:], probabilitiesssss_m_k[1:])
    plt.xlabel('Classes')
    plt.ylabel('Probability')
    plt.title('Class Probabilities')
    plt.savefig(save_path3)
    which = [m_n,n_k,m_k]
    final = max(set(which), key=which.count)

    if which.count(final) == 1:
        final = "other"
    result={
        "disease": final,
        "n_k": f'{n_k}:{str(round(max(probabilitiesssss_n_k)))}%',
        "m_n": f'{m_n}:{str(round(max(probabilitiesssss_m_n)))}%',
        "m_k": f'{m_k}:{str(round(max(probabilitiesssss_m_k)))}%',
    }

    #return final,n_k+":"+str(max(probabilitiesssss_n_k)),m_n+":"+str(max(probabilitiesssss_m_n)),m_k+":"+str(max(probabilitiesssss_m_k))
    return result
#print(predict_images("melanoma/ISIC_0012099.jpg","result/1","result/2","result/3"))


# # 调用函数预测文件夹中的所有图像（修改为你的文件夹路径）
# folder_path = 'melanoma'
# folder_class = 'm'
#
# correct, total, correct_3, correct_2, final, correct_final = predict_all_images_in_folder(folder_path, folder_class)
#
# print(f"Correct: {correct_final}/{total}")
# print(f"Accuracy: {correct_final/total * 100:.2f}%")
#
# folder_path = 'seborrheic_keratosis'
# folder_class = 'k'
#
# correct, total, correct_3, correct_2, final, correct_final = predict_all_images_in_folder(folder_path, folder_class)
#
# print(f"Correct: {correct_final}/{total}")
# print(f"Accuracy: {correct_final/total * 100:.2f}%")
#
# folder_path = 'nevus'
# folder_class = 'n'
#
# correct, total, correct_3, correct_2, final, correct_final = predict_all_images_in_folder(folder_path, folder_class)
#
# print(f"Correct: {correct_final}/{total}")
# print(f"Accuracy: {correct_final/total * 100:.2f}%")

# best_modelresnet18_success_1
# Correct: 17/30
# Accuracy: 56.67%
# Correct: 22/42
# Accuracy: 52.38%
# Correct: 44/78
# Accuracy: 56.41%

# best_modelresnet18_success5_v1
# Correct: 12/30
# Accuracy: 40.00%
# Correct: 20/42
# Accuracy: 47.62%
# Correct: 56/78
# Accuracy: 71.79%



