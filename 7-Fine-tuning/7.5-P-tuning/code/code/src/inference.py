# -*- coding: utf-8 -*-
"""
P-tuning 推理脚本

加载训练好的 P-tuning 模型进行预测和文本生成。
"""

import os
import json
import argparse
from typing import List, Dict, Union, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from .ptuning_model import PTuningModel
from ..config import P TuningConfig


class P TuningInference:
    """
    P-tuning 推理类
    
    支持：
    - 文本分类预测
    - 文本生成
    - 批量推理
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = None
    ):
        """
        初始化推理器
        
        Args:
            model_path: 模型目录路径
            device: 推理设备（默认自动选择）
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"加载模型：{model_path}")
        print(f"使用设备：{self.device}")
        
        # 加载配置
        config_path = os.path.join(model_path, "ptuning_config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 加载模型
        self.model = PTuningModel.from_pretrained(model_path, device=self.device)
        self.model.eval()
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        
        # 设置 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载标签映射（如果存在）
        self.label_map = self._load_label_map()
        
        print(f"模型加载完成")
        print(f"  基础模型：{self.config['model_name']}")
        print(f"  虚拟词元数量：{self.config['num_virtual_tokens']}")
        print(f"  任务类型：{self.config['task_type']}")
    
    def _load_label_map(self) -> Optional[Dict[int, str]]:
        """加载标签映射"""
        label_map_path = os.path.join(self.model_path, "label_map.json")
        if os.path.exists(label_map_path):
            with open(label_map_path, 'r', encoding='utf-8') as f:
                # 将字符串键转换为整数
                return {int(k): v for k, v in json.load(f).items()}
        return None
    
    @torch.no_grad()
    def predict(
        self,
        text: Union[str, List[str]],
        return_probabilities: bool = False
    ) -> Union[int, List[int], Dict]:
        """
        文本分类预测
        
        Args:
            text: 输入文本（单个或列表）
            return_probabilities: 是否返回概率
        
        Returns:
            预测标签或包含标签和概率的字典
        """
        if self.config['task_type'] != 'classification':
            raise ValueError("predict() 仅适用于分类任务")
        
        # 支持批量输入
        if isinstance(text, str):
            text = [text]
            single_input = True
        else:
            single_input = False
        
        # 分词
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.config.get('max_seq_length', 128),
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # 前向传播
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        logits = outputs.logits
        
        # 计算概率
        probabilities = F.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)
        
        # 整理结果
        results = []
        for i in range(len(text)):
            pred = predictions[i].item()
            prob = probabilities[i].cpu().numpy()
            
            result = {
                'text': text[i],
                'label': pred,
                'label_name': self.label_map.get(pred, str(pred)) if self.label_map else str(pred),
                'probabilities': prob.tolist()
            }
            results.append(result)
        
        if single_input:
            return results[0] if return_probabilities else results[0]['label']
        else:
            return results if return_probabilities else [r['label'] for r in results]
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        num_return_sequences: int = 1,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = None,
        do_sample: bool = True,
    ) -> Union[str, List[str]]:
        """
        文本生成
        
        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成 token 数
            num_return_sequences: 返回序列数量
            temperature: 采样温度（>1 增加随机性，<1 减少随机性）
            top_p: nucleus sampling 参数
            top_k: top-k sampling 参数
            do_sample: 是否采样（False 则使用贪婪解码）
        
        Returns:
            生成的文本
        """
        if self.config['task_type'] != 'causal_lm':
            raise ValueError("generate() 仅适用于 causal_lm 任务")
        
        # 分词
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # 生成
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
        )
        
        # 解码
        # 注意：跳过虚拟词元部分
        num_virtual_tokens = self.config['num_virtual_tokens']
        
        generated_texts = []
        for i in range(generated_ids.shape[0]):
            # 跳过 prompt 和虚拟词元，只解码新生成的部分
            generated_text = self.tokenizer.decode(
                generated_ids[i][input_ids.shape[1]:],  # 跳过原始输入
                skip_special_tokens=True
            )
            generated_texts.append(generated_text)
        
        if num_return_sequences == 1:
            return generated_texts[0]
        else:
            return generated_texts
    
    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 16
    ) -> List[int]:
        """
        批量预测（适用于大量数据）
        
        Args:
            texts: 文本列表
            batch_size: 批大小
        
        Returns:
            预测标签列表
        """
        all_predictions = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_preds = self.predict(batch_texts)
            all_predictions.extend(batch_preds)
        
        return all_predictions
    
    def get_confidence(self, text: str) -> float:
        """
        获取预测置信度
        
        Args:
            text: 输入文本
        
        Returns:
            预测概率（最高类别的概率）
        """
        result = self.predict(text, return_probabilities=True)
        return max(result['probabilities'])
    
    def explain_prediction(self, text: str) -> Dict:
        """
        解释预测结果
        
        Args:
            text: 输入文本
        
        Returns:
            包含预测详情和解释的字典
        """
        result = self.predict(text, return_probabilities=True)
        
        explanation = {
            'text': text,
            'predicted_label': result['label_name'],
            'confidence': max(result['probabilities']),
            'all_probabilities': {
                str(i): prob for i, prob in enumerate(result['probabilities'])
            }
        }
        
        return explanation


def create_demo_inference():
    """创建演示用的推理示例"""
    print("="*60)
    print("P-tuning 推理演示")
    print("="*60)
    
    # 示例 1: 情感分类
    print("\n【示例 1: 情感分类】")
    print("假设我们已经训练了一个情感分类模型...")
    
    # 模拟预测结果
    sample_texts = [
        "这部电影太棒了，演员表演精彩！",
        "非常糟糕的体验，完全不推荐。",
        "一般般，没有特别的感觉。"
    ]
    
    print("\n输入文本:")
    for text in sample_texts:
        print(f"  - {text}")
    
    print("\n预测结果（模拟）:")
    print("  - '这部电影太棒了...' → 正面 (置信度：0.95)")
    print("  - '非常糟糕的体验...' → 负面 (置信度：0.98)")
    print("  - '一般般...' → 中性 (置信度：0.60)")
    
    # 示例 2: 文本生成
    print("\n【示例 2: 文本生成】")
    print("假设我们有一个生成模型...")
    
    prompts = [
        "今天天气真好，",
        "人工智能正在改变",
        "推荐一部电影："
    ]
    
    print("\n输入提示:")
    for prompt in prompts:
        print(f"  - {prompt}")
    
    print("\n生成结果（模拟）:")
    print("  - '今天天气真好，' → '适合出去散步，阳光明媚，微风不燥。'")
    print("  - '人工智能正在改变' → '我们的生活方式，从智能家居到自动驾驶，AI 无处不在。'")
    print("  - '推荐一部电影：' → '《肖申克的救赎》，这是一部关于希望和自由的经典之作。'")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="P-tuning 推理脚本")
    parser.add_argument("--model_path", type=str, required=True, help="模型目录路径")
    parser.add_argument("--text", type=str, default=None, help="输入文本（分类任务）")
    parser.add_argument("--prompt", type=str, default=None, help="输入提示（生成任务）")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="最大生成 token 数")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="返回序列数量")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度")
    parser.add_argument("--device", type=str, default=None, help="推理设备")
    
    args = parser.parse_args()
    
    # 检查模型路径
    if not os.path.exists(args.model_path):
        print(f"错误：模型路径不存在：{args.model_path}")
        return
    
    # 创建推理器
    try:
        inferencer = P TuningInference(args.model_path, device=args.device)
    except Exception as e:
        print(f"加载模型失败：{e}")
        return
    
    # 执行推理
    if args.text:
        # 分类任务
        print(f"\n输入文本：{args.text}")
        result = inferencer.predict(args.text, return_probabilities=True)
        print(f"预测标签：{result['label_name']}")
        print(f"置信度：{max(result['probabilities']):.4f}")
        print(f"所有概率：{result['probabilities']}")
    
    elif args.prompt:
        # 生成任务
        print(f"\n输入提示：{args.prompt}")
        generated = inferencer.generate(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=args.num_return_sequences,
            temperature=args.temperature
        )
        print(f"生成结果：{generated}")
    
    else:
        # 无输入，显示帮助
        print("\n请提供 --text（分类）或 --prompt（生成）参数")
        print("\n示例:")
        print("  分类：python inference.py --model_path ./output/best_model --text '这部电影很好看'")
        print("  生成：python inference.py --model_path ./output/best_model --prompt '今天天气真好'")


if __name__ == "__main__":
    # 如果没有提供参数，运行演示
    import sys
    if len(sys.argv) == 1:
        create_demo_inference()
    else:
        main()
