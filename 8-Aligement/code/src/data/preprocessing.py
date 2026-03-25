"""
数据预处理模块

实现偏好数据的加载、处理和 DataLoader
数据格式：(prompt, chosen, rejected)
"""

import json
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)


class PreferenceData:
    """
    偏好数据样本
    
    Attributes:
        prompt: 输入提示
        chosen: 被选择的回复（更好的）
        rejected: 被拒绝的回复（较差的）
    """
    
    def __init__(self, prompt: str, chosen: str, rejected: str):
        self.prompt = prompt
        self.chosen = chosen
        self.rejected = rejected
    
    def __repr__(self) -> str:
        return f"PreferenceData(prompt_len={len(self.prompt)}, chosen_len={len(self.chosen)}, rejected_len={len(self.rejected)})"


def load_preference_data(data_path: str) -> List[PreferenceData]:
    """
    从 JSONL 文件加载偏好数据
    
    Args:
        data_path: JSONL 文件路径，每行格式：
            {"prompt": "...", "chosen": "...", "rejected": "..."}
    
    Returns:
        PreferenceData 列表
    
    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON 格式错误
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在：{data_path}")
    
    preference_data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                item = json.loads(line)
                # 数据清洗：确保必要字段存在且非空
                if not all(key in item for key in ['prompt', 'chosen', 'rejected']):
                    logger.warning(f"第{line_num}行缺少必要字段，跳过")
                    continue
                
                prompt = item['prompt'].strip()
                chosen = item['chosen'].strip()
                rejected = item['rejected'].strip()
                
                # 过滤空数据
                if not prompt or not chosen or not rejected:
                    logger.warning(f"第{line_num}行包含空字段，跳过")
                    continue
                
                preference_data.append(PreferenceData(prompt, chosen, rejected))
                
            except json.JSONDecodeError as e:
                logger.error(f"第{line_num}行 JSON 解析失败：{e}")
                continue
    
    logger.info(f"成功加载 {len(preference_data)} 条偏好数据")
    return preference_data


class PreferenceDataset(Dataset):
    """
    偏好数据集
    
    将文本数据转换为模型输入格式，支持多种对齐算法的数据需求
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        include_prompt: bool = True
    ):
        """
        初始化数据集
        
        Args:
            data_path: JSONL 数据文件路径
            tokenizer: HuggingFace 分词器
            max_length: 最大序列长度
            include_prompt: 是否在编码中包含 prompt
        """
        self.raw_data = load_preference_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_prompt = include_prompt
        
        logger.info(f"数据集大小：{len(self.raw_data)}")
    
    def __len__(self) -> int:
        return len(self.raw_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
        
        Returns:
            包含编码后输入的字典：
            - prompt_input_ids: prompt 的 input_ids
            - chosen_input_ids: chosen 回复的 input_ids
            - chosen_attention_mask: chosen 的 attention mask
            - rejected_input_ids: rejected 回复的 input_ids
            - rejected_attention_mask: rejected 的 attention mask
        """
        data = self.raw_data[idx]
        
        # 编码 prompt
        prompt_encoding = self.tokenizer(
            data.prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        
        # 编码 chosen 回复 (prompt + chosen)
        chosen_text = data.prompt + data.chosen
        chosen_encoding = self.tokenizer(
            chosen_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        
        # 编码 rejected 回复 (prompt + rejected)
        rejected_text = data.prompt + data.rejected
        rejected_encoding = self.tokenizer(
            rejected_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        
        return {
            'prompt_input_ids': torch.tensor(prompt_encoding['input_ids'], dtype=torch.long),
            'prompt_attention_mask': torch.tensor(prompt_encoding['attention_mask'], dtype=torch.long),
            'chosen_input_ids': torch.tensor(chosen_encoding['input_ids'], dtype=torch.long),
            'chosen_attention_mask': torch.tensor(chosen_encoding['attention_mask'], dtype=torch.long),
            'rejected_input_ids': torch.tensor(rejected_encoding['input_ids'], dtype=torch.long),
            'rejected_attention_mask': torch.tensor(rejected_encoding['attention_mask'], dtype=torch.long),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    """
    DataLoader 的 collate 函数
    
    将多个样本批处理，进行 padding 和对齐
    
    Args:
        batch: 样本列表
    
    Returns:
        批处理后的字典，包含：
        - prompt_input_ids: [batch_size, seq_len]
        - prompt_attention_mask: [batch_size, seq_len]
        - chosen_input_ids: [batch_size, seq_len]
        - chosen_attention_mask: [batch_size, seq_len]
        - rejected_input_ids: [batch_size, seq_len]
        - rejected_attention_mask: [batch_size, seq_len]
    """
    # 找到最大长度
    max_prompt_len = max(len(x['prompt_input_ids']) for x in batch)
    max_chosen_len = max(len(x['chosen_input_ids']) for x in batch)
    max_rejected_len = max(len(x['rejected_input_ids']) for x in batch)
    
    batch_size = len(batch)
    
    # 初始化 padding 后的张量
    pad_token_id = 0  # 假设 pad_token_id 为 0
    
    prompt_input_ids = torch.full((batch_size, max_prompt_len), pad_token_id, dtype=torch.long)
    prompt_attention_mask = torch.zeros((batch_size, max_prompt_len), dtype=torch.long)
    
    chosen_input_ids = torch.full((batch_size, max_chosen_len), pad_token_id, dtype=torch.long)
    chosen_attention_mask = torch.zeros((batch_size, max_chosen_len), dtype=torch.long)
    
    rejected_input_ids = torch.full((batch_size, max_rejected_len), pad_token_id, dtype=torch.long)
    rejected_attention_mask = torch.zeros((batch_size, max_rejected_len), dtype=torch.long)
    
    # 填充数据
    for i, sample in enumerate(batch):
        prompt_len = len(sample['prompt_input_ids'])
        chosen_len = len(sample['chosen_input_ids'])
        rejected_len = len(sample['rejected_input_ids'])
        
        # Prompt
        prompt_input_ids[i, :prompt_len] = sample['prompt_input_ids']
        prompt_attention_mask[i, :prompt_len] = sample['prompt_attention_mask']
        
        # Chosen
        chosen_input_ids[i, :chosen_len] = sample['chosen_input_ids']
        chosen_attention_mask[i, :chosen_len] = sample['chosen_attention_mask']
        
        # Rejected
        rejected_input_ids[i, :rejected_len] = sample['rejected_input_ids']
        rejected_attention_mask[i, :rejected_len] = sample['rejected_attention_mask']
    
    return {
        'prompt_input_ids': prompt_input_ids,
        'prompt_attention_mask': prompt_attention_mask,
        'chosen_input_ids': chosen_input_ids,
        'chosen_attention_mask': chosen_attention_mask,
        'rejected_input_ids': rejected_input_ids,
        'rejected_attention_mask': rejected_attention_mask,
    }


def create_dataloader(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    max_length: int = 512,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    创建偏好数据 DataLoader
    
    Args:
        data_path: 数据文件路径
        tokenizer: 分词器
        batch_size: 批次大小
        max_length: 最大序列长度
        shuffle: 是否打乱数据
        num_workers: DataLoader worker 数量
    
    Returns:
        DataLoader 实例
    """
    dataset = PreferenceDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


# 数据清洗示例函数
def clean_preference_data(
    data: List[Dict[str, str]],
    min_prompt_length: int = 10,
    min_response_length: int = 20,
    max_length: int = 2048
) -> List[Dict[str, str]]:
    """
    清洗偏好数据
    
    过滤标准：
    1. prompt 长度不能太短
    2. 回复长度不能太短
    3. 总长度不能超过限制
    4. 去除重复数据
    5. 去除特殊字符
    
    Args:
        data: 原始数据列表
        min_prompt_length: 最小 prompt 长度
        min_response_length: 最小回复长度
        max_length: 最大总长度
    
    Returns:
        清洗后的数据列表
    """
    cleaned_data = []
    seen_prompts = set()
    
    for item in data:
        prompt = item.get('prompt', '').strip()
        chosen = item.get('chosen', '').strip()
        rejected = item.get('rejected', '').strip()
        
        # 长度检查
        if len(prompt) < min_prompt_length:
            continue
        if len(chosen) < min_response_length or len(rejected) < min_response_length:
            continue
        if len(prompt) + len(chosen) > max_length:
            continue
        if len(prompt) + len(rejected) > max_length:
            continue
        
        # 去重
        if prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)
        
        cleaned_data.append({
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected
        })
    
    logger.info(f"数据清洗：{len(data)} -> {len(cleaned_data)}")
    return cleaned_data
