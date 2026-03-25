# 参考资料来源

## 原始材料
- **来源文件**: D:\Projects\LLM-tutorial-AllinOne\7-Fine-tuning\7.4-Prefix-tuning\README.md
- **内容**: Prefix-tuning 基础介绍、特点、数学表达、架构设计

## 核心论文
- **论文标题**: Prefix-Tuning: Optimizing Continuous Prompts for Generation
- **作者**: Xiang Lisa Li, Percy Liang
- **发表年份**: 2021
- **链接**: https://arxiv.org/abs/2101.00190
- **关键贡献**:
  - 提出连续可微的软提示概念
  - 在 Transformer 每层注入前缀向量
  - 实现与全量微调相当的效果，仅用 0.1% 可训练参数

## 技术要点整理

### 1. Prefix-tuning 特点
- 冻结预训练模型参数，不进行更新
- 通过添加可训练的、针对特定任务的前缀来适应不同任务
- 存储时只需保存前缀部分，大大降低了微调的成本

### 2. 与上下文学习的区别
| 特性 | GPT-3 上下文学习 | Prefix-tuning |
|------|-----------------|---------------|
| 提示类型 | 离散提示（人工设计） | 连续可微提示 |
| 优化方式 | 不可优化 | 端到端梯度下降 |
| 稳定性 | 对措辞敏感 | 更稳定 |
| 表达能力 | 受限于词汇表 | 连续空间，更灵活 |

### 3. 数学表达
- 原始公式：$Y = WX$
- Prefix-tuning：$Y = W'X$，其中 $W' = [W_p; W]$

### 4. 重参数化技巧
$$
h_{i}=\left\{\begin{array}{l l}{P_{\theta}[i,:]=\mathrm{MLP}_{\theta}(P_{\theta}^{\prime}[i,:]),i\in P_{\mathrm{idx}}}\\ {\mathrm{LM}_{\phi}(z_{i},h_{<i}),i\in P_{\mathrm{idx}}}\end{array}\right.
$$

### 5. 不同架构的适配
- **自回归模型（GPT）**: `[PREFIX; x; y]`
- **编码器 - 解码器模型（BART）**: `[PREFIX; x; PREFIX'; y]`

### 6. 实验对比
| 方法 | 可训练参数 | 效果 |
|------|-----------|------|
| 全量微调 | 100% | 基准 |
| Embedding-only 前缀 | ~0.1% | 略优于全量微调 |
| Full 前缀（每层） | ~0.1-1% | 明显优于 Embedding-only |

### 7. 优势总结
1. 采用连续可微的软提示设计
2. 性能优于 Adapter 等其他 PEFT 方法
3. 能达到与全量微调相当的效果

### 8. 局限性
1. 训练难度较高
2. 数据空间受限（前缀占用序列长度）

## 参考图片
- 传统微调与前缀微调对比图
- 自回归模型和编码器 - 解码器模型的前缀结构图

## 其他参考资料
- HuggingFace PEFT 文档：https://huggingface.co/docs/peft
- Prefix-tuning 官方代码：https://github.com/XiangLi1999/PrefixTuning

---
*本资料由 Searcher Agent 整理，用于 Prefix-tuning 文章撰写*
