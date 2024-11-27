# 归一化

在 LLM 中，归一化是一种数据处理技术，通过将输入特征缩放到统一的尺度上，来提高模型的泛化能力和训练效率。

1.  Pre-Norm 和 Post-Norm

![Alt text](pre_post_norm.png)

**2. Pre-Norm**

Sublayer 表示自注意力层或前馈神经网络层。

![](https://img-blog.csdnimg.cn/img_convert/d1e5a602faacc78542914df5fa7c565d.png)

**3. Post-Norm**

![](https://img-blog.csdnimg.cn/img_convert/12200b316ce5244da06682bfbe3f0fd4.png)

**4. LayerNorm**

![](https://img-blog.csdnimg.cn/img_convert/e97db5ea237639fd260c3415d10e96d9.png)

![](https://img-blog.csdnimg.cn/img_convert/1f7c1d45e20c405bc25c543701b23492.png)

![](https://img-blog.csdnimg.cn/img_convert/c4038b907b80bac062fe7d0bcf2b8cf6.png)

**5. RMSNorm**

RMSNorm 省略了 LayerNorm 中平均值μ的计算，只基于均方根进行缩放。

![](https://img-blog.csdnimg.cn/img_convert/0af0e846509438f63865affeaf8bec60.png)