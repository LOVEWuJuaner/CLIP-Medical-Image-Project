# Experiment_2

## 目标
下载肺部CT或者X光开源数据集
在CLIP上实现分类

## 过程记录
我用原来猫狗分类的代码，结合config.yaml将参数改成了肺部的感染有无，一共正样本负样本各一百张左右，但是CLIP全识别成了感染，正在思考怎么调整？我试试改prompt？
这是这次的prompt： 
 - "a normal chest x-ray"
 - "a chest x-ray with pneumonia"

## 决定将实验二拿来做各种提示词的效果对比

第一次：
思考：开始尝试
提示词：
 - "a normal chest x-ray"
 - "a chest x-ray with pneumonia"
准确率：50%
分析：
  概率左低右高，几乎全是90%多的概率判定为肺炎，无效

第二次
思考：把两个都改了一下
提示词：
  - "a healthy chest x-ray"
  - "a medical image showing pneumonia"
准确率：54.05%
分析：
  开始出现正例的判断，但效果仍极差

第三次
思考：
    1.提示词是两个，判断互不影响，我觉得得先控制一个变量，好观察变化，重复两轮，最后将最好得两个提示词拼在一起，这一轮先改动NORMAL的提示词；（又想了想，其实没必要，反正互不干扰，跑一次相当于跑两次）

    2.观察前两次的数据，我发现第二次在第一次的基础上，判为NORMAL的概率均变大了，判为PNEUMONIA的概率均变小了，这不是一个好的迹象
提示词：
  - "a clear chest x-ray with no disease"
  - "a chest x-ray showing lung infection"
准确率：50%
分析：
  总体仍是左低右高，但没第一次那么悬殊

第四次
思考：
    换用医学情景的描述
提示词：
  - "a radiology chest x-ray of a healthy patient"
  - "a radiology image showing pneumonia in the lungs"
准确率：53.38%
分析：
  左右差距又减小，双方大体往50%靠拢，仍未能显现出靠谱的可能
