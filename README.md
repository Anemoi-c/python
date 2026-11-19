本项目为 Python 程序设计课程 Project 1 的代码，使用纯 NumPy 复现 OpenCLIP ViT-B/32 图像编码器的前向计算全过程。全程不依赖 PyTorch 等深度学习框架，输出结果与课程提供的参考实现数值对齐。
环境要求
• Python 3.9 及以上版本
• 依赖包：numpy、Pillow
安装依赖：
bash


pip install -r
 requirements.txt

运行说明
准备工作
将课程提供的 image_encoder.npz 权重文件放入项目下的 weights 文件夹中。
数值校验
提供了分阶段校验脚本，可以逐个验证每个环节的输出是否正确：
bash
# 校验资源文件完整性
python check.py 
--stage
 assets

# 分阶段验证（调试时建议按顺序跑）
python check.py 
--stage
 preprocess
python check.py 
--stage
 patches
python check.py 
--stage
 tokens
python check.py 
--stage
 output

# 一次性跑完所有校验
python check.py 
--stage
 all

提取图像特征
使用入口脚本可以对指定图片提取特征向量：
bash


python run.py --images test.jpg --output
 tokens.npz

实现内容
基于课程给出的代码框架，补全了 6 处核心功能：
1. 图像分块：将输入的 [B,3,224,224] 图像切分为 49 个 32×32 补丁并按顺序展平
2. 多头拆分：将特征维度拆分为 12 个注意力头，调整张量维度以适配矩阵运算
3. 多头注意力：完成 QK 点积、缩放、注意力权重计算、值加权求和，以及最终的输出投影
4. Transformer 块：实现 Pre-LN 结构的 Transformer 块，包含注意力和 MLP 两条残差支路
5. 嵌入层：完成补丁的线性投影，拼接 CLS 令牌，并叠加可学习的位置编码
6. 输出头：提取 CLS 位置的特征，经过后归一化与投影层，最终输出单位长度的 512 维特征
