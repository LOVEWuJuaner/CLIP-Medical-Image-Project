import os  # Operating System（操作系统）它就是 Python 自带的、用来跟你的电脑文件夹、文件打交道的工具。
import torch
import clip
from PIL import Image
import yaml
from pathlib import Path  # Pathlib 可以更方便地处理路径问题，自动兼容 Windows/Linux

cnt = 0
right = 0
image_index = 1

# 选择实验配置文件
config_path = Path(__file__).parent.parent / "experiments" / "experiment_2" / "config.yaml"

# 读取配置
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
# 脚本目录
BASE_DIR = Path(__file__).parent.parent  # src 的父目录 = 项目根目录
# 使用配置
data_path = BASE_DIR / config["data"]["path"]  # 自动拼接成绝对路径
image_size = config["data"]["image_size"]  # 图片预处理尺寸
batch_size = config["inference"]["batch_size"]  # 批量大小（可以暂时没用）
texts = config["input_texts"]  # 文本描述列表
output_range = config["output_range"]

print("数据路径:", data_path)
print("batch size:", batch_size)
print("文本:", texts)

device = "cuda" if torch.cuda.is_available() else "cpu"  # 定义在哪运行
model, preprocess = clip.load("ViT-B/32", device=device)  # 加载CLIP模型，图片预处理工具（改尺寸，转张量）
text_tokens = clip.tokenize(texts).to(device)  # 文本 token 化

# 遍历每个类别
for real_label, class_name in enumerate(output_range):
    # 拼接文件夹路径
    # 注意：class_name 对应 data_path 下的文件夹名
    folder_path = data_path / class_name
    if not folder_path.exists():
        print(f"警告：文件夹不存在 {folder_path}")
        continue

    # 遍历文件夹中的每一张图片
    for img_path in folder_path.iterdir():
        if not img_path.is_file():
            continue
        # 打开+预处理图片
        image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():  # 关闭梯度运算，只推理，不训练
            logits_per_image, _ = model(image, text_tokens)  # 计算相似度
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()  # 把相似度打分转为概率，归一化0-1，转到cpu，将张量变回numpy数组

        # 拿到预测结果
        pred_idx = probs[0].argmax()  # 找 probs[0] 最大数的下标
        print(f"{image_index} {texts[0]}: {probs[0][0]:.4f}  {texts[1]}: {probs[0][1]:.4f} | 真实：{class_name}  预测：{output_range[pred_idx]}")
        cnt += 1
        image_index += 1
        if pred_idx == real_label:
            right += 1

if cnt > 0:
    print(f"识别准确率为 {right/cnt*100:.2f}%")
else:
    print("没有找到图片进行推理！")