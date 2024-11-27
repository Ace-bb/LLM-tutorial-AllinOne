**多头注意力(Multi-Head Attention)**

因为一段文字可能蕴含了比如情感维度、时间维度、逻辑维度等很多维度的特征，为了能从不同的维度抓住输入信息的重点，chatGPT使用了多头注意力机制(multi-head attention)。

而所谓多头注意力，简单说就是把输入序列投影为多组不同的Query，Key，Value，并行分别计算后，再把各组计算的结果合并作为最终的结果，通过使用多头注意力机制，ChatGPT可以更好地捕获来自输入的多维度特征，提高模型的表达能力和泛化能力，并减少过拟合的风险。

**多头注意力机制的目的是为了从多个维度捕捉提取更多的特征，从多个“头”得到不同的Self-Attention Score，提高模型表现。**

**首先放一张论文原文中的多头注意力机制的架构（Multi-Head Attention），可以看到（V,K,Q）三个矩阵通过h个线性变换（Linear），分别得到h组（V,K,Q）矩阵，每一组（V,K,Q）经过Attention计算，得到h个Attention Score并进行拼接（Concat），最后通过一个线性变换得到输出，其维度与输入词向量的维度一致，其中h就是多头注意力机制的“头数”。**

![图片](https://i-blog.csdnimg.cn/blog_migrate/5fd02f549e1d4e0dc33f8ef36202011d.png)

**下图为更直观的表示论文中的计算过程**，以输入词“**X**=[‘图’, ’书’, ’馆’]”为例，句子长度为3，词向量的维度为4，这里将词向量分为2个头，线性变换后得到2组（V0,K0,Q0）和（V1,K1,Q1），每组（V,K,Q）进行Self-Attention计算得到两个Score即（Z0和Z1），将Z0和Z1进行拼接Concat后进行线性变换得到输出向量Z，其维度与输入矩阵维度相同。

**ps：多头注意力机制代码实现和论文里的模式不一样哦（详见下面）！！！**

![图片](https://i-blog.csdnimg.cn/blog_migrate/68129bde294a368df1ee06215c10f05e.png)

**下图是代码实现的过程，不同于论文，代码中对（V,K,Q）进行一次线性变换，然后在特征维度上进行h次分割（在代码中就是通过矩阵转置transpose和维度变换torch.view）后得到h组（V,K,Q），分别计算Self-Attention Score后进行Concat拼接**（同样的通过一系列的transpose和torch.view），最后通过线性变换得到最后的输出。

![图片](https://i-blog.csdnimg.cn/blog_migrate/31072e381ede01bc815832ca3ef712c8.png)

最后附一张代码截图，馆长的代码可能写的不太规范和严谨**，仅根据论文实现主要功能**，**并没有体现Dropout和mask等全面的功能，主要是为了通过代码实现来更好的理解Transformer中多头注意力机制，**仅供学习交流参考，在实战中的代码需要进一步完善。

**ps：多头注意力中K、Q、V的线性层具有相同输入和输出尺寸是一种常见且实用的设计选择**

![](https://i-blog.csdnimg.cn/direct/426274cc093343c8b6300f24c2aa766e.png)

![](https://i-blog.csdnimg.cn/direct/66b35b9a1d2d4e3d88ccbb91d2642c69.png)

![图片](https://i-blog.csdnimg.cn/blog_migrate/721e994cc968af0dd7b5c608bb9760df.png)

```
`

1.  import torch
2.  import torch.nn as nn
3.  import torch.nn.functional as F

5.  # 这个代码中省略了输入矩阵x转变为qkv的过程！！！

7.  class MultiHeadAttention(nn.Module):
8.      def __init__(self, heads, d_model):
9.          super(MultiHeadAttention, self).__init__()
10.          self.d_model = d_model
11.          self.heads = heads
12.          # 定义K, Q, V的权重矩阵
13.          # 多头注意力中K、Q、V的线性层具有相同输入和输出尺寸是一种常见且实用的设计选择！！！
14.          self.k_linear = nn.Linear(d_model, d_model)
15.          self.q_linear = nn.Linear(d_model, d_model)
16.          self.v_linear = nn.Linear(d_model, d_model)
17.          # 分头后的维度
18.          self.d_token = d_model // heads
19.          # 定义输出权重矩阵
20.          self.out = nn.Linear(d_model, d_model)

22.      def forward(self, q, k, v):
23.          # 计算batch大小
24.          batch = q.size(0)

26.          # 线性变换后的Q, K, V，然后分割成多个头
27.          k = self.k_linear(k).view(batch, -1, self.heads, self.d_token)
28.          q = self.q_linear(q).view(batch, -1, self.heads, self.d_token)
29.          v = self.v_linear(v).view(batch, -1, self.heads, self.d_token)

31.          # 转置调整维度，以计算注意力分数
32.          k = k.transpose(1, 2)  # 形状变为 [batch, heads, seq_len, d_token]
33.          q = q.transpose(1, 2)
34.          v = v.transpose(1, 2)

36.          # 计算自注意力分数
37.          scores = self.attention(q, k, v, self.d_token)

39.          # 调整形状以进行拼接
40.          scores = scores.transpose(1, 2).contiguous().view(batch, -1, self.d_model)

42.          # 通过输出权重矩阵进行线性变换
43.          output = self.out(scores)
44.          return output

46.      @staticmethod
47.      def attention(q, k, v, d_token):
48.          # 计算注意力分数 (q @ k^T) / sqrt(d_token)
49.          scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(d_token)
50.          # 应用softmax归一化（沿着最后一个维度（dim=-1））
51.          attn = F.softmax(scores, dim=-1)
52.          # 计算加权的V
53.          output = torch.matmul(attn, v)
54.          return output

`![](https://csdnimg.cn/release/blogv2/dist/pc/img/newCodeMoreWhite.png)
```

![](https://i-blog.csdnimg.cn/direct/8e4105cd1ca644d9abd00c839e869241.png)

代码解析：

        k = self.k_linear(k).view(batch, -1, self.heads, self.d_token)  
        q = self.q_linear(q).view(batch, -1, self.heads, self.d_token)  
        v = self.v_linear(v).view(batch, -1, self.heads, self.d_token)

![](https://i-blog.csdnimg.cn/direct/549c51f25cb443f5910953d27fe6af02.png)

![](https://i-blog.csdnimg.cn/direct/941b88fb371e4d5d94240d39f5ce9624.png)

    # 转置调整维度，以计算注意力分数  
        k = k.transpose(1, 2)  # 形状变为 [batch, heads, seq_len, d_token]  
        q = q.transpose(1, 2)  
        v = v.transpose(1, 2)

![](https://i-blog.csdnimg.cn/direct/eb248ca6773841a0959ec8654c49d685.png)

scores.transpose(1, 2).contiguous().view(batch, -1, self.d_model)

![](https://i-blog.csdnimg.cn/direct/a50ba1db5f87494c8510d5fc4ae29e2f.png)

![](https://i-blog.csdnimg.cn/direct/4fb1ef3aa6594f95950847a50bea486e.png)

scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(d_token)

![](https://i-blog.csdnimg.cn/direct/ee1c7511c52f455fa21f88c3d3200afa.png)

**ps：实际在Transformer中多头注意力机制：在计算出注意力结果并经过线性层和dropout之后，还需要和Q进行短接！！！**![](https://i-blog.csdnimg.cn/direct/1a9e9bc243ae44a7afae8d8898376e44.png)

```
 `2.  # 多头注意力计算层
3.  class MultiHead(torch.nn.Module):
4.      def __init__(self):
5.          super().__init__()
6.          self.fc_Q = torch.nn.Linear(32, 32)# 就是那三个初始化QKV的参数矩阵
7.          self.fc_K = torch.nn.Linear(32, 32)
8.          self.fc_V = torch.nn.Linear(32, 32)

10.          self.out_fc = torch.nn.Linear(32, 32)

12.          self.norm = torch.nn.LayerNorm(normalized_shape=32, elementwise_affine=True)

14.          self.dropout = torch.nn.Dropout(p=0.1)

16.      def forward(self, Q, K, V, mask):
17.          # b句话,每句话50个词,每个词编码成32维向量
18.          # Q,K,V = [b, 50, 32]
19.          b = Q.shape[0]

21.          # 克隆Q：保留下原始的Q,后面要做短接用
22.          clone_Q = Q.clone() # 克隆副本，并不共享空间！

24.          # 规范化（论文是放在后面，但实际都是放在前面。经过广泛论证这样效果会更好，能更好地帮助模型收敛！）
25.          Q = self.norm(Q)
26.          K = self.norm(K)
27.          V = self.norm(V)

29.          # 线性运算,维度不变
30.          # [b, 50, 32] -> [b, 50, 32]
31.          K = self.fc_K(K)
32.          V = self.fc_V(V)
33.          Q = self.fc_Q(Q)

35.          # 拆分成多个头
36.          # b句话,每句话50个词,每个词编码成32维向量,4个头,每个头分到8维向量
37.          # [b, 50, 32] -> [b, 4, 50, 8]
38.          Q = Q.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)
39.          K = K.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)
40.          V = V.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)

42.          # 计算注意力
43.          # [b, 4, 50, 8] -> [b, 50, 32]
44.          score = attention(Q, K, V, mask)

46.          # 计算输出,维度不变
47.          # [b, 50, 32] -> [b, 50, 32]
48.          score = self.dropout(self.out_fc(score))# dropout防止过拟合

50.          # 短接
51.          score = clone_Q + score
52.          return score`![](https://csdnimg.cn/release/blogv2/dist/pc/img/newCodeMoreWhite.png)
```

![](https://i-blog.csdnimg.cn/direct/9c51bd5896d64464b6df21d925e3ac00.png)