import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        
        self.query=nn.Linear(embed_dim, embed_dim)
        self.key=nn.Linear(embed_dim, embed_dim)
        self.value=nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
        
    def forward(self,X):
        batch_size, seq_len, embed_dim = X.size()
         #线性映射
        q = self.query(X)
        k = self.key(X)
        v = self.value(X)
        print("q1 shape:", q.shape)
        print("k1 shape:", k.shape)
        #[batch_size, seq_len, embed_dim]变为[batch_size, seq_len, num_heads, head_dim]
        #transpose(1, 2) 调换了 seq_len 和 num_heads 的维度[batch_size, num_heads, seq_len, head_dim]
        q=q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        print("q shape:", q.shape)
        print("k shape:", k.shape)

        
        #最核心的，计算点积
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        
        # 首先计算 k 的转置，k.transpose(-2, -1) 将 k 的最后两个维度调换，形状变为 [batch_size, num_heads，head_dim, seq_len]。
        # 然后计算 q 和转置后的 k 的点积，torch.matmul(q, k.transpose(-2, -1)) 结果形状为 [batch_size, num_heads, seq_len, seq_len]。
        # 最后除以 sqrt(head_dim) 进行缩放，这是为了稳定梯度，防止点积结果过大。
        attn_output = torch.matmul(attn_weights, v)
        #v: [batch_size, num_heads, seq_len, head_dim]结果
        # attn_output为：[batch_size, num_heads, seq_len, head_dim]。
#        这一步是加权求和，将每个位置的值向量 v 根据注意力权重进行加权求和。
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out(attn_output)
        return output
# 简单的词汇表和分词器
def simple_tokenizer(sentence):
    word_to_index = {'this': 1, 'is': 2, 'an': 3, 'example': 4, 'sentence': 5}
    tokens = sentence.lower().split()
    return [word_to_index.get(word, 0) for word in tokens]

# 函数：将句子编码为向量
def encode_sentence(sentence, tokenizer, max_len=10):
    tokens = tokenizer(sentence)
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    else:
        tokens = tokens + [0] * (max_len - len(tokens))
    return torch.tensor(tokens, dtype=torch.long)

# 示例数据
sentence = "this is an example sentence"
vocab_size = 6  # 假设词汇表大小，包括 0
embedding_dim = 512#输入每个x1的维度
max_len = 10
# 创建一个嵌入层
embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
# 将句子编码为输入向量X
encoded_sentence = encode_sentence(sentence, simple_tokenizer, max_len)
# 嵌入句子
embedded_sentence = embedding_layer(encoded_sentence)
#上面步骤是构造一个#尺度为max_len乘embedding_dim 的向量


num_heads = 8
# 自定义多头注意力机制
attention_layer = MultiHeadAttention(embed_dim=embedding_dim, num_heads=num_heads)
attention_output = attention_layer(embedded_sentence.unsqueeze(0))  # 添加 batch 维度

print("Encoded Sentence:", encoded_sentence)
print("Embedded Sentence Shape:", embedded_sentence.shape)
print("Attention Output Shape:", attention_output.shape)