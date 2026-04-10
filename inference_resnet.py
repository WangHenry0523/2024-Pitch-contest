import torch.nn as nn
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
accuracy=0
times=0
#三個類別
classes=["nevus","melanoma","seborrheic_keratosis"]

#載入模型
# model = torchvision.models.mobilenet_v2()
# num_ftrs = model.classifier[1].in_features
# model.classifier[1] = nn.Linear(num_ftrs, 5)

model=torchvision.models.resnet18()
num_ftrs=model.fc.in_features
model.fc=nn.Linear(num_ftrs,3)

checkpoint = torch.load('best_single_isic_model_resnet18.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
model.eval()

#圖像預處理步驟
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

#推理函數
def predict(image_path, svae_path, svae_path2, svae_path3):
    global times, accuracy, classes
    max=0
    times+=1
    probability=[]
    #處理圖片
    input_image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(input_image)
    input_tensor = input_tensor.unsqueeze(0)
    #推理
    with torch.no_grad():
        output = model(input_tensor)
        print("* 模型原始輸出:", output)
        output_max=output.argmax(1)
        print("* pridict={}".format(classes[output_max.item()]))
        if classes[output_max.item()]=='melanoma':
            accuracy+=1
        probabilities = torch.nn.functional.softmax(output[0], dim=0)*100
        print(f'* {probabilities}')
    #產生機率
    for i in range(len(classes)):
        print(f'* {classes[i]}: {probabilities[i].item():.1f}%')
        probability.append(probabilities[i].item())
        if probabilities[i].item()>max:
            max=probabilities[i].item()
        
    #畫出圖片
    print(f'* {probability}')
    try:
        plt.figure()
        plt.bar(classes[1:], probability[1:])
        plt.xlabel('Classes')
        plt.ylabel('Probability')
        plt.title('Class Probabilities')
        plt.savefig(save_path)
    except Exception as e:
        print(f'* {e}')
    final = str(round(max,1)) +'%'
    result = {
        'disease' : str(classes[output_max.item()]).capitalize(),
        'accuracy' : final
    }
    return result

#temp_result = predict('/home/jetson/final_v5/static/uploads/2024-08-05_14-40-06.jpg')
