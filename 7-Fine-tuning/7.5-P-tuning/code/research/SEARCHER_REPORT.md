# Searcher Agent 任务完成报告

**任务：** P-tuning 技术参考资料搜索与整理

**完成时间：** 2026-03-16 16:17 GMT+8

**执行 Agent：** Searcher Agent 🔍

---

## 一、任务完成情况

### ✅ 已完成工作

1. **核心论文信息收集**
   - ✅ 原始 P-tuning 论文《GPT Understands, Too》(arxiv:2103.10385)
   - ✅ P-tuning V2 论文 (arxiv:2110.07602)
   - ✅ Prefix-Tuning 论文 (arxiv:2101.00190)
   - ✅ Prompt Tuning 论文 (arxiv:2104.08691)

2. **官方代码仓库确认**
   - ✅ THUDM/P-tuning（原始实现）
   - ✅ THUDM/P-tuning-v2（V2 实现）
   - ✅ HuggingFace PEFT 库（集成实现）

3. **技术细节确认**
   - ✅ 虚拟词元数量：50-100（默认 50）
   - ✅ 提示编码器架构：LSTM(512) + MLP
   - ✅ 学习率：1e-3 ~ 1e-4
   - ✅ 参数量：约 0.01%-0.1%

4. **P-tuning 与 V2 区分**
   - ✅ 明确两种方法的核心区别
   - ✅ 标注本文讲解的是原始 P-tuning
   - ✅ 提供对比表格和选择建议

5. **代码实现参考**
   - ✅ 官方仓库代码结构
   - ✅ HuggingFace PEFT 用法
   - ✅ 自定义完整实现示例

---

## 二、输出文件清单

### 主文档

**文件路径：** `articles/P-tuning 技术详解/research/sources.md`

**文件大小：** 14,041 bytes

**内容概要：**
- 核心论文列表（4 篇关键论文）
- 官方代码仓库（3 个主要仓库）
- 技术博客与教程（3 个推荐来源）
- 关键技术参数确认表
- 应用场景与局限性分析
- 三种方法对比总结（Prefix-Tuning vs P-tuning vs Prompt Tuning）
- 资料可信度评级

---

### 关键片段（Snippets）

#### 1. 论文摘要片段
**文件：** `research/snippets/paper-abstract.md` (4,222 bytes)

**内容：**
- 原始论文标题、作者、机构
- 完整摘要翻译与解读
- 核心方法概述
- 关键实验结果
- 引用格式

#### 2. P-tuning vs V2 对比
**文件：** `research/snippets/ptuning-vs-v2.md` (6,530 bytes)

**内容：**
- 核心区别速查表
- 提示添加位置对比（图解）
- 参数量对比计算
- 适用任务范围
- 性能对比数据
- 实现复杂度对比
- 选择建议

#### 3. 技术原理详解
**文件：** `research/snippets/technical-details.md` (11,680 bytes)

**内容：**
- 整体架构图（ASCII）
- 虚拟词元详解（数学表示 + 代码）
- 提示编码器架构（LSTM+MLP 详细参数）
- 嵌入组合实现
- 训练流程（参数冻结、训练循环）
- 超参数建议表
- 数学原理（目标函数、梯度流）

#### 4. 代码实现参考
**文件：** `research/snippets/code-references.md` (14,960 bytes)

**内容：**
- 官方仓库代码结构
- HuggingFace PEFT 用法
- 自定义完整实现（PromptEncoder + PTuningModel）
- 数据加载示例
- 推理示例
- 关键实现细节（虚拟词元位置、损失计算）
- 保存和加载方法
- 调试技巧
- 常见问题解答

---

## 三、关键发现与确认

### 3.1 重要技术参数

| 参数 | 确认值 | 来源 |
|------|--------|------|
| 虚拟词元数量 | 50（默认），20-100 可调 | 论文 + 官方代码 |
| LSTM 隐藏层 | 512 | 论文 + 官方代码 |
| 学习率 | 1e-3 | 官方代码 |
| 批大小 | 16-32 | 官方实验设置 |
| 训练轮数 | 50-200（根据任务） | P-tuning v2 仓库 |
| 优化器 | AdamW (weight_decay=0.01) | 官方代码 |

### 3.2 P-tuning 与 V2 的核心区别

**最重要区别：提示添加位置**

- **原始 P-tuning：** 仅在输入层添加虚拟词元
- **P-tuning V2：** 在每一层 Transformer 都添加提示（Deep Prompt Tuning）

**性能影响：**
- V2 能处理序列标注等困难任务
- V2 在小模型上表现更好
- V2 参数量更多（0.1%-3% vs 0.01%）

**本文范围：** 讲解**原始 P-tuning**，不涵盖 V2

### 3.3 与其他方法对比

**vs Prefix-Tuning：**
- P-tuning 更简单（仅输入层）
- Prefix-Tuning 每层都添加提示
- P-tuning 针对 NLU，Prefix-Tuning 针对生成

**vs Prompt Tuning：**
- P-tuning 有 LSTM+MLP 编码器
- Prompt Tuning 直接学习虚拟词元
- P-tuning 在小模型上收敛更快

---

## 四、资料来源可信度

| 来源类型 | 可信度 | 采用情况 |
|---------|--------|---------|
| 原始论文（arxiv:2103.10385） | ⭐⭐⭐⭐⭐ | 核心依据 |
| 官方代码仓库（THUDM/P-tuning） | ⭐⭐⭐⭐⭐ | 代码参考 |
| 相关论文（Prefix/Prompt Tuning） | ⭐⭐⭐⭐⭐ | 对比参考 |
| HuggingFace PEFT | ⭐⭐⭐⭐⭐ | 实现参考 |
| Lilian Weng 博客 | ⭐⭐⭐⭐ | 辅助理解 |
| 社区技术文章 | ⭐⭐⭐ | 交叉验证后采用 |

---

## 五、注意事项

### ⚠️ 重要提醒

1. **P-tuning 与 V2 不混淆**
   - 所有资料已明确标注是原始 P-tuning 还是 V2
   - sources.md 中有专门对比章节
   - 代码参考基于原始 P-tuning

2. **虚拟词元数量灵活调整**
   - 默认 50，但需根据任务调整
   - 简单任务 20-30，复杂任务 80-100
   - 已在技术参数表中说明

3. **HuggingFace PEFT 实现差异**
   - PEFT 实现的是 Prompt Tuning（无 LSTM 编码器）
   - 与原始 P-tuning 有区别
   - 已提供自定义完整实现作为补充

4. **模型规模影响**
   - P-tuning 在大模型（>1B）上效果更好
   - 小模型建议考虑 V2 或全量微调
   - 已在局限性部分说明

---

## 六、后续建议

### 给 Writer Agent 的建议

1. **文章结构：** 按照 outline.md 的五大模块撰写
2. **技术准确性：** 关键参数参考 sources.md 中的确认值
3. **代码实现：** 使用 code-references.md 中的完整示例
4. **V2 区分：** 在文章适当位置明确说明讲解的是原始 P-tuning
5. **图表建议：** 
   - 整体架构图（参考 technical-details.md）
   - P-tuning vs V2 对比表（参考 ptuning-vs-v2.md）

### 给 Coder Agent 的建议

1. **代码结构：** 参考 code-references.md 中的项目结构
2. **核心实现：** PromptEncoder 类使用 technical-details.md 中的实现
3. **超参数：** 使用 sources.md 中确认的默认值
4. **测试验证：** 确保代码可运行，参数量计算正确

---

## 七、文件存储位置

```
articles/P-tuning 技术详解/
└── research/
    ├── sources.md                    # 主文档（14KB）
    └── snippets/
        ├── paper-abstract.md         # 论文摘要（4KB）
        ├── ptuning-vs-v2.md          # V2 对比（6.5KB）
        ├── technical-details.md      # 技术原理（11.7KB）
        └── code-references.md        # 代码参考（14.9KB）
```

**总计：** 约 52KB 参考资料

---

## 八、任务完成确认

✅ **所有搜索重点已完成：**
1. ✅ 原始论文关键信息
2. ✅ P-tuning 与 V2 的区别
3. ✅ 技术原理（提示词编码器、LSTM+MLP）
4. ✅ 代码实现（官方仓库、HuggingFace、自定义）
5. ✅ 应用场景与局限性

✅ **所有输出要求已满足：**
1. ✅ 结构化参考资料列表
2. ✅ 重要技术参数确认
3. ✅ 代码参考来源
4. ✅ 与 Prefix-Tuning、Prompt Tuning 对比

✅ **注意事项已遵守：**
- ✅ 优先权威来源（论文、官方文档）
- ✅ 标注每个来源的可信度
- ✅ P-tuning 与 V2 信息明确区分

---

**Searcher Agent 任务完成。** 

所有参考资料已整理完毕，可供 Writer Agent 和 Coder Agent 使用。

---

*报告生成时间：2026-03-16 16:17 GMT+8*
*Searcher Agent 🔍*
