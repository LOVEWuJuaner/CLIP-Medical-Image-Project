import os  # Operating System（操作系统）它就是 Python 自带的、用来跟你的电脑文件夹、文件打交道的工具。
import torch
import clip
from PIL import Image
cnt = 0
right = 0

# 获取当前脚本文件的绝对路径
script_path = os.path.abspath(__file__)
# 获取脚本所在目录（即 src 目录）
src_dir = os.path.dirname(script_path)
# 从 src 目录向上跳一级到项目根目录，再拼接数据集路径
DATASET_ROOT = os.path.join(src_dir, "..", "data", "sample", "dog&cat_set")
# 标准化路径（把 ../ 解析成真实路径）
DATASET_ROOT = os.path.normpath(DATASET_ROOT)

device = "cuda" if torch.cuda.is_available() else "cpu"  # 定义在哪运行
model, preprocess = clip.load("ViT-B/32", device=device)  # 加载CLIP模型，图片预处理工具（改尺寸，转张量）
text = clip.tokenize(["a cat", "a dog"]).to(device)
# 遍历两个文件夹：cat、dog
for real_label, class_name in enumerate(["cat", "dog"]):
    # 拼接文件夹路径
    folder_path = os.path.join(DATASET_ROOT, class_name)
    # 遍历这个文件夹里 每一张图片
    for img_name in os.listdir(folder_path):
        # 单张图片完整路径
        img_path = os.path.join(folder_path, img_name)
        #  打开+预处理图片
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

        with torch.no_grad():  # 关闭梯度运算，只推理，不训练
            logits_per_image,_= model(image, text)  # 计算相似度
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()  # 把相似度打分转为概率，归一化0-1，转到cpu，将张量变回numpy数组
        # 拿到预测结果
        pred_idx = probs[0].argmax()  # 找probs【0】最大数的下标
        print(f"cat: {probs[0][0]:.4f}  dog: {probs[0][1]:.4f} | 真实：{class_name}  预测：{['cat', 'dog'][pred_idx]}")
        cnt += 1
        if pred_idx == real_label:
            right += 1
print(f"识别准确率为{right/cnt*100:.2f}%")