import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import math
import random

# Constants
CONTEXT_WINDOW_SIZE = 128000
MAX_OUTPUT_TOKENS_PREVIEW = 32768
MAX_OUTPUT_TOKENS_MINI = 65536

# Set random seeds for reproducibility
torch.manual_seed(0)
random.seed(0)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Ensure x has the correct shape (batch_size, seq_len, d_model)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        elif x.dim() == 4:
            x = x.squeeze(2)  # Remove extra dimension if present
        
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x

class O1Model(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, is_mini=False):
        super(O1Model, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer_layers = nn.ModuleList([TransformerBlock(d_model, nhead) for _ in range(num_layers)])
        self.completion_decoder = nn.Linear(d_model, vocab_size)
        self.reasoning_decoder = nn.Linear(d_model, vocab_size)
        self.value_head = nn.Linear(d_model, 1)
        self.subtask_head = nn.Linear(d_model, 1)
        self.is_mini = is_mini
        self.max_reasoning_tokens = 1000

    def forward(self, src, reasoning_tokens=None, generate_reasoning=True):
        if src.dim() == 1:
            src = src.unsqueeze(0)
        elif src.dim() == 3:
            src = src.squeeze(1)
        
        if src.size(1) == 0:
            print(f"Warning: Empty input tensor in forward pass. Shape: {src.shape}")
            batch_size = src.size(0)
            return torch.zeros(batch_size, 1, self.vocab_size), torch.zeros(batch_size, 1, self.vocab_size), torch.zeros(batch_size, 1)
        
        src = self.embed(src)
        if reasoning_tokens is not None:
            reasoning_embeddings = self.embed(reasoning_tokens)
            src = torch.cat([src, reasoning_embeddings], dim=1)
        
        src = self.pos_encoder(src)
        
        for layer in self.transformer_layers:
            src = layer(src)
        
        completion_logits = self.completion_decoder(src)
        values = self.value_head(src).squeeze(-1)
        
        if generate_reasoning:
            reasoning_logits = self.reasoning_decoder(src)
            return completion_logits, reasoning_logits, values
        else:
            return completion_logits, values

    def generate_completion(self, input_ids, max_new_tokens, num_paths=3):
        max_tokens = MAX_OUTPUT_TOKENS_MINI if self.is_mini else MAX_OUTPUT_TOKENS_PREVIEW
        max_new_tokens = min(max_new_tokens, max_tokens)
        
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        elif input_ids.dim() == 3:
            input_ids = input_ids.squeeze(1)
        
        paths = []
        for _ in range(num_paths):
            generated = input_ids.clone()
            reasoning_tokens = torch.tensor([], dtype=torch.long, device=input_ids.device)
            completion_tokens = []
            subtasks = []
            
            for _ in range(max_new_tokens):
                if generated.size(1) + reasoning_tokens.size(0) >= CONTEXT_WINDOW_SIZE:
                    break
                
                completion_logits, reasoning_logits, values = self(generated, reasoning_tokens)
                
                if completion_logits.numel() == 0:
                    print(f"Warning: completion_logits is empty. Input shape: {generated.shape}")
                    break
                
                next_token_logits = completion_logits[:, -1, :]
                next_token = self.sample_token(next_token_logits)
                
                reasoning_token = self.sample_token(reasoning_logits[:, -1, :])
                reasoning_tokens = torch.cat([reasoning_tokens, reasoning_token.unsqueeze(0)])
                
                if reasoning_tokens.size(0) > self.max_reasoning_tokens:
                    reasoning_tokens = reasoning_tokens[-self.max_reasoning_tokens:]
                
                last_hidden = self.embed(generated[:, -1])
                subtask_prob = torch.sigmoid(self.subtask_head(last_hidden))
                if subtask_prob > 0.5:
                    subtask = self.generate_subtask(generated, reasoning_tokens)
                    subtasks.append(subtask)
                    generated = torch.cat([generated, torch.tensor([[vocab['<subtask>']]]).to(generated.device)], dim=1)
                else:
                    generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
                    completion_tokens.append(next_token.item())
                
                if self.should_revise_reasoning():
                    generated, reasoning_tokens = self.revise_reasoning(generated, reasoning_tokens)
                
                if next_token.item() == vocab['<eos>']:
                    break
            
            paths.append((completion_tokens, reasoning_tokens.tolist(), subtasks))
        
        if not paths:
            print("Warning: No valid paths generated")
            return [], [], []
        
        rewards = [self.compute_reward(p[0], p[1], p[2]) for p in paths]
        best_path = paths[rewards.index(max(rewards))]
        
        return best_path[0], best_path[1], best_path[2]

    def sample_token(self, logits, temperature=0.7):
        probs = F.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)

    def add_reasoning_token(self, token):
        self.reasoning_buffer.append(token)
        if len(self.reasoning_buffer) > self.max_reasoning_tokens:
            self.reasoning_buffer.pop(0)

    def should_revise_reasoning(self):
        # Implement logic to decide if reasoning should be revised
        return random.random() < 0.1  # 10% chance of revision for demonstration

    def revise_reasoning(self, generated, reasoning_tokens):
        # Implement logic to revise reasoning
        # For demonstration, we'll just remove the last few tokens from both
        return generated[:, :-5], reasoning_tokens[:-5]

    def generate_subtask(self, context, reasoning_tokens):
        subtask_tokens = []
        for _ in range(20):  # Max subtask length
            logits, _, _ = self(context, reasoning_tokens)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            subtask_tokens.append(next_token.item())
            context = torch.cat([context, next_token.unsqueeze(1)], dim=1)
            if next_token.item() == vocab['<eos>']:
                break
        return subtask_tokens

    def compute_reward(self, completion_tokens, reasoning_tokens, subtasks):
        completion_reward = len(completion_tokens) * 0.1
        reasoning_reward = len(set(reasoning_tokens)) * 0.2
        subtask_reward = len(subtasks) * 0.5
        coherence_reward = self.compute_coherence(completion_tokens)
        process_reward = self.compute_process_reward(reasoning_tokens)
        return completion_reward + reasoning_reward + subtask_reward + coherence_reward + process_reward

    def compute_coherence(self, tokens):
        # Simple coherence check (can be made more sophisticated)
        return sum(1 for i in range(len(tokens)-1) if tokens[i] + 1 == tokens[i+1]) * 0.1

    def compute_process_reward(self, reasoning_tokens):
        # Implement a more sophisticated process reward
        unique_tokens = len(set(reasoning_tokens))
        return unique_tokens * 0.1  # Reward diverse reasoning

class PPO:
    def __init__(self, model, optimizer, clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
        self.model = model
        self.optimizer = optimizer
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def compute_advantages(self, rewards, values, gamma=0.99, lambda_=0.95):
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        # Make sure to only iterate through the valid range
        for t in reversed(range(len(rewards))):
            if t + 1 < len(values):
                delta = rewards[t] + gamma * values[t + 1] - values[t]
            else:
                delta = rewards[t] - values[t]
                
            advantages[t] = delta + gamma * lambda_ * last_advantage
            last_advantage = advantages[t]
        
        returns = advantages + values[:len(advantages)]
        return advantages, returns

    def update(self, states, actions, old_log_probs, rewards, old_values):
        # Reshape states if necessary
        if states.dim() == 2:
            batch_size, seq_len = states.shape
            states = states.unsqueeze(0)  # Add a dimension to make it [1, batch_size, seq_len]
        else:
            num_steps, batch_size, seq_len = states.shape
        
        # Flatten other tensors
        actions_flat = actions.view(-1)
        old_log_probs_flat = old_log_probs.view(-1)
        advantages, returns = self.compute_advantages(rewards, old_values)
        advantages_flat = advantages.view(-1)
        returns_flat = returns.view(-1)
        
        for _ in range(5):  # PPO epochs
            logits, _, values = self.model(states.view(-1, seq_len))
            
            # Focus on the logits of the last token in the sequence
            next_token_logits = logits[:, -1, :]
            new_probs = F.softmax(next_token_logits, dim=-1)
            dist = Categorical(new_probs)
            
            # Ensure actions_flat matches the shape of new_probs
            actions_flat_truncated = actions_flat[:new_probs.size(0)]
            old_log_probs_flat_truncated = old_log_probs_flat[:new_probs.size(0)]
            advantages_flat_truncated = advantages_flat[:new_probs.size(0)]
            returns_flat_truncated = returns_flat[:new_probs.size(0)]
            
            # Calculate new log probabilities
            new_log_probs = dist.log_prob(actions_flat_truncated)
            
            # Calculate probability ratio
            ratio = torch.exp(new_log_probs - old_log_probs_flat_truncated)
            surr1 = ratio * advantages_flat_truncated
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_flat_truncated
            
            # Compute losses
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Extract the value of the last token in each sequence
            values_last = values[:, -1].view(-1)
            critic_loss = nn.MSELoss()(values_last, returns_flat_truncated)
            
            entropy = dist.entropy().mean()
            
            # Total loss
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# Enhanced vocabulary
vocab = {
    '<pad>': 0, '<sos>': 1, '<eos>': 2, 'Step:': 3, '+': 4, '-': 5, '*': 6, '/': 7, '=': 8,
    '0': 9, '1': 10, '2': 11, '3': 12, '4': 13, '5': 14, '6': 15, '7': 16, '8': 17, '9': 18,
    'if': 19, 'then': 20, 'else': 21, 'greater': 22, 'less': 23, 'equal': 24,
    'Calculate': 25, 'the': 26, 'sum': 27, 'of': 28, 'and': 29,
    'difference': 30, 'between': 31, 'product': 32, 'quotient': 33,
    'First,': 34, 'Next,': 35, 'Finally,': 36, 'result': 37, 'is': 38,
    '<subtask>': 39  # New token for subtask generation
}
vocab_size = len(vocab)
inv_vocab = {v: k for k, v in vocab.items()}

def tokenize(text):
    return [vocab.get(token, vocab['<pad>']) for token in text.strip().split()]

def detokenize(indices):
    return ' '.join([inv_vocab.get(idx, ' ') for idx in indices])

# Update the compute_reward function
def compute_reward(state, target_result):
    # 这一行代码从 `state` 中提取出每个样本的最后一个生成 token（假设 `state` 是一个包含多个时间步的序列数据）。`state[:, -1]` 取出最后一个时间步的生成 token。
    generated_tokens = state[:, -1].cpu().numpy() 
    rewards = []
    for tokens in generated_tokens:
        try:
            # 对于每个生成的 `tokens`，调用 `detokenize` 函数将 tokens 转换为生成的文本
            generated_text = detokenize(tokens)
            # 检查生成的文本中是否包含 "result is"，这是生成模型用于标识答案或推理结果的标志。如果存在，则截取 "result is" 后面的部分作为结果字符串。
            # 提取 "result is" 后面的字符串并去掉首尾空白字符。
            if "result is" in generated_text:
                result_str = generated_text.split("result is")[-1].strip()
                # 将提取出来的结果字符串转换为数字。如果 `result_str` 是整数，转为 `int` 类型；否则，转为 `float` 类型。
                result = int(result_str) if result_str.isdigit() else float(result_str)
                """
                结果级奖励：根据生成的结果与目标结果之间的差距来分配奖励：
                - 如果生成结果与目标结果的差距小于 1e-6（考虑到浮动误差），则奖励为 1.0，表示完全正确。
                - 如果差距小于 5，则奖励为 0.5，表示较为接近。
                - 如果差距小于 10，奖励为 0.2，表示有一定接近。
                - 如果差距大于 10，则奖励为 -0.2，表示较为远离正确答案。
                """
                if abs(result - target_result) < 1e-6:  # Allow for small floating-point differences
                    rewards.append(1.0)
                elif abs(result - target_result) < 5:  # Close answer
                    rewards.append(0.5)
                elif abs(result - target_result) < 10:  # Somewhat close answer
                    rewards.append(0.2)
                else:
                    rewards.append(-0.2)
            else:
                # 如果生成的文本中没有包含 "result is"，则说明模型没有给出完整的答案或没有给出有效的结果，在这种情况下返回中立奖励 0.0，表示没有好的表现，但也不做惩罚。
                rewards.append(0.0)  # Neutral reward for incomplete answers
        except:
            # 如果在处理过程中发生任何错误（例如，detokenize 过程失败或无法正确转换），则会进入 except 语句块，返回一个负奖励 -0.5，表示惩罚模型输出不正确的结果或无效的输出。
            rewards.append(-0.5)  # Penalize malformed outputs
    return torch.tensor(rewards)

# Generate arithmetic problems
def generate_arithmetic_problem():
    operations = ['+', '-', '*', '/']
    op = random.choice(operations)
    
    while True:
        if op in ['+', '-']:
            a, b = random.randint(1, 100), random.randint(1, 100)
        else:
            a, b = random.randint(1, 10), random.randint(1, 10)
        
        if op == '+':
            result = a + b
            problem = f"Calculate the sum of {a} and {b}"
        elif op == '-':
            result = a - b
            problem = f"Calculate the difference between {a} and {b}"
        elif op == '*':
            result = a * b
            problem = f"Calculate the product of {a} and {b}"
        else:
            if b != 0:  # Avoid division by zero
                result = a // b
                problem = f"Calculate the quotient of {a} and {b}"
            else:
                continue  # Try again if b is zero
        
        if problem and result:
            return problem, result

# Generate reasoning chain
def generate_reasoning_chain(problem, result):
    words = problem.split()
    operation = words[3]  # "sum", "difference", "product", or "quotient"
    
    if operation == "sum":
        a, b = map(int, words[-3::2])
        chain = f"Step: First, we identify the numbers: {a} and {b}. "
        chain += f"Next, we add these numbers: {a} + {b}. "
        chain += f"Finally, we get the result: The sum is {result}."
    elif operation == "difference":
        a, b = map(int, words[-3::2])
        chain = f"Step: First, we identify the numbers: {a} and {b}. "
        chain += f"Next, we subtract the second number from the first: {a} - {b}. "
        chain += f"Finally, we get the result: The difference is {result}."
    elif operation == "product":
        a, b = map(int, words[-3::2])
        chain = f"Step: First, we identify the numbers: {a} and {b}. "
        chain += f"Next, we multiply these numbers: {a} * {b}. "
        chain += f"Finally, we get the result: The product is {result}."
    else:  # quotient
        a, b = map(int, words[-3::2])
        chain = f"Step: First, we identify the numbers: {a} and {b}. "
        chain += f"Next, we divide the first number by the second: {a} / {b}. "
        chain += f"Finally, we get the result: The quotient is {result}."
    
    return chain

# Modify collect_trajectories to use arithmetic problems
def collect_trajectories(model, batch_size):
    states = []
    actions = []
    rewards = []
    log_probs = []
    values = []

    max_state_length = 40

    for _ in range(batch_size):
        problem, result = generate_arithmetic_problem()
        reasoning_chain = generate_reasoning_chain(problem, result)
        
        input_ids = torch.tensor([tokenize(problem)])
        target_ids = torch.tensor([tokenize(reasoning_chain)])
        
        state = input_ids
        action_sequence = torch.full((1, max_state_length), vocab['<pad>'], dtype=torch.long)

        for t in range(max_state_length):
            if state.size(1) > max_state_length:
                state = state[:, :max_state_length]
            elif state.size(1) < max_state_length:
                padding = torch.full((1, max_state_length - state.size(1)), vocab['<pad>'], dtype=state.dtype)
                state = torch.cat([state, padding], dim=1)

            with torch.no_grad():
                logits, _, value = model(state)
                probs = F.softmax(logits[:, -1, :], dim=-1)
                dist = Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            action_sequence[0, t] = action.item()
            log_probs.append(log_prob)
            values.append(value[:, -1])

            state = torch.cat([state[:, :-1], action.unsqueeze(1)], dim=1)

            reward = compute_reward(state, result)
            rewards.append(reward)

            if action.item() == vocab['<eos>']:
                break

        states.append(state)
        actions.append(action_sequence)

    states = torch.cat(states, dim=0)
    actions = torch.cat(actions, dim=0)
    rewards = torch.cat(rewards, dim=0)
    log_probs = torch.cat(log_probs, dim=0)
    values = torch.cat(values, dim=0)

    return states, actions, rewards, log_probs, values

# Update the training function
def train_o1_model(model, optimizer, num_epochs, batch_size):
    ppo = PPO(model, optimizer)
    
    for epoch in range(num_epochs):
        # Generate a batch of arithmetic problems
        states, actions, rewards, old_log_probs, values = collect_trajectories(model, batch_size)
        
        # Supervised learning step
        sl_loss = supervised_finetuning_loss(model, (states, actions))
        optimizer.zero_grad()
        sl_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
        # Reinforcement learning step
        ppo.update(states, actions, old_log_probs, rewards, values)
    
        # Evaluation and logging
        if epoch % 10 == 0:
            metrics = evaluate_model(model, batch_size)
            log_metrics(metrics, epoch)

        print(f'Epoch {epoch} completed')
            
        # Dynamic curriculum learning
        if epoch % 50 == 0:
            adjust_problem_difficulty(epoch)

def log_metrics(metrics, epoch):
    print(f"Epoch {epoch} Metrics: {metrics}")

def supervised_finetuning_loss(model, batch):
    states, actions = batch
    logits, _ = model(states, generate_reasoning=False)
    
    # Reshape logits to [batch_size * sequence_length, vocab_size]
    batch_size, seq_length, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size)
    
    # Reshape actions to [batch_size * sequence_length]
    target_ids = actions.view(-1)
    
    # Ensure logits and target_ids have the same length
    min_length = min(logits.size(0), target_ids.size(0))
    logits = logits[:min_length]
    target_ids = target_ids[:min_length]
    
    # Compute loss only on non-padded tokens
    non_pad_mask = target_ids != vocab['<pad>']
    logits = logits[non_pad_mask]
    target_ids = target_ids[non_pad_mask]
    
    loss = F.cross_entropy(logits, target_ids)
    return loss

# Update evaluation function
def evaluate_model(model, batch_size):
    model.eval() # 将模型切换到评估模式，关闭诸如 dropout 或 batch normalization 等只在训练阶段启用的机制，从而使得评估时模型的行为更加稳定。
    total_reward = 0 # 用于累加所有有效样本的奖励值。
    valid_samples = 0 # 用于计数有效样本的数量，这有助于在最后计算平均奖励时避免除以零的错误。
    # 表示在此代码块内，不会进行梯度计算。这是因为评估过程中不需要进行反向传播，关闭梯度计算可以节省内存和计算资源。
    with torch.no_grad():
        # 遍历给定的批次大小 `batch_size`。对于每一个样本，生成一个算术问题并评估模型的推理能力。
        for _ in range(batch_size):
            try:
                # 会随机生成一个算术问题，并计算出其正确的答案 `result`。
                problem, result = generate_arithmetic_problem()
                input_ids = torch.tensor([tokenize(problem)])
                
                # 这部分代码确保生成的 `input_ids` 不为空。如果为空，说明 `tokenize` 过程中可能出现了问题，因此会跳过该样本，并打印警告信息。
                if input_ids.numel() == 0:
                    print(f"Warning: Empty input tensor for problem: {problem}")
                    continue
                
                # 调用模型生成一个推理链，模型根据给定的算术问题 `input_ids` 推理并生成 `completion_tokens`（即推理链的标记），`reasoning_tokens` 和 `subtasks` 分别可能是推理过程中的中间步骤和子任务（具体依赖于模型的实现）。
                completion_tokens, reasoning_tokens, subtasks = model.generate_completion(input_ids, max_new_tokens=50) # `max_new_tokens=50` 设置了推理链的最大长度。
                
                # 如果模型成功生成了推理链（`completion_tokens` 不为空），则使用 `compute_reward` 函数计算该推理链的奖励。奖励是根据生成的推理链与目标结果 `result` 的匹配程度来计算的。
                if completion_tokens:
                    reward = compute_reward(torch.tensor([completion_tokens]), result)
                    total_reward += reward.item() #  奖励被累加到 `total_reward` 中，并且有效样本计数 `valid_samples` 增加。
                    valid_samples += 1
                else:
                    print(f"Warning: Empty output for problem: {problem}")
            except Exception as e:
                print(f"Error during evaluation: {e}")
    
    model.train()  # 将模型重新设置为训练模式
    avg_reward = total_reward / valid_samples if valid_samples > 0 else 0 # 计算平均奖励
    return {"average_reward": avg_reward, "valid_samples": valid_samples}

def adjust_problem_difficulty(epoch):
    # Implement dynamic difficulty adjustment based on model performance
    global problem_difficulty
    if epoch < 100:
        problem_difficulty = "easy"
    elif epoch < 300:
        problem_difficulty = "medium"
    else:
        problem_difficulty = "hard"

if __name__ == "__main__":
    # Model parameters
    d_model = 128
    nhead = 8
    num_layers = 4
    dropout = 0.1

    # Initialize the model
    model = O1Model(vocab_size, d_model, nhead, num_layers)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    # Training parameters
    num_epochs = 500
    batch_size = 64

    # Train the model
    train_o1_model(model, optimizer, num_epochs, batch_size)

    # Save the model
    torch.save(model.state_dict(), "o1_model.pth")