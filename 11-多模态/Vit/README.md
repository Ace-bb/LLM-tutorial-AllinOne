# VIT论文
[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

## 摘要
虽然 Transformer 架构已成为自然语言处理任务的事实标准，但其在计算机视觉中的应用仍然有限。 在视觉领域，注意力机制要么与卷积网络一起使用，要么用于替换卷积网络的某些组件，同时保留其整体结构。 我们表明，这种对 CNN 的依赖性不是必需的，直接应用于图像块序列的纯 Transformer 在图像分类任务上可以表现得非常好。 当在大量数据上进行预训练并迁移到多个中等规模或小型图像识别基准（ImageNet、CIFAR-100、VTAB 等）时，Vision Transformer (ViT) 与最先进的卷积网络相比取得了优异的结果，同时训练所需的计算资源明显减少。

![alt text](./_img/vit_frame.png)


## 推荐阅读
1. Transformer官方解读：[https://hugging-face.cn/docs/transformers/model_doc/vit](https://hugging-face.cn/docs/transformers/model_doc/vit)