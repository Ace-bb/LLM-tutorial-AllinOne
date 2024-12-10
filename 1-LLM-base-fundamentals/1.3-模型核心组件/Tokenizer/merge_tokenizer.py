import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"
from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
from pathlib import Path


def merge_tokenizer(llama_tokenizer_dir,tr_sp_model_file):
    # load
    llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
    tr_sp_model = spm.SentencePieceProcessor()
    tr_sp_model.Load(tr_sp_model_file)
    
    llama_spm = sp_pb2_model.ModelProto()
    llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
    tr_spm = sp_pb2_model.ModelProto()
    tr_spm.ParseFromString(tr_sp_model.serialized_model_proto())
    
    # print number of tokens
    print(len(llama_tokenizer),len(tr_sp_model))
    print(llama_tokenizer.all_special_tokens)
    print(llama_tokenizer.all_special_ids)
    print(llama_tokenizer.special_tokens_map)
    
    ## Add tr tokens to LLaMA tokenizer
    llama_spm_tokens_set=set(p.piece for p in llama_spm.pieces)
    print(len(llama_spm_tokens_set))
    print(f"Before:{len(llama_spm_tokens_set)}")
    for p in tr_spm.pieces:
        piece = p.piece
        if piece not in llama_spm_tokens_set:
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            new_p.score = 0
            llama_spm.pieces.append(new_p)
    print(f"New model pieces: {len(llama_spm.pieces)}")
    
    ## Save
    output_sp_dir = 'merged_tokenizer_sp'
    output_hf_dir = 'merged_tokenizer_hf' # the path to save Chinese-LLaMA tokenizer
    os.makedirs(output_sp_dir,exist_ok=True)
    with open(output_sp_dir+'/tr_llama.model', 'wb') as f:
        f.write(llama_spm.SerializeToString())
    tokenizer = LlamaTokenizer(vocab_file=output_sp_dir+'/tr_llama.model')
    
    tokenizer.save_pretrained(output_hf_dir)
    print(f"tr-LLaMA tokenizer has been saved to {output_hf_dir}")

    return Path(output_hf_dir), Path(output_sp_dir)


# Test
#llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
#tr_llama_tokenizer = LlamaTokenizer.from_pretrained(output_hf_dir)
#print(tokenizer.all_special_tokens)
#print(tokenizer.all_special_ids)
#print(tokenizer.special_tokens_map)
#text='''白日依山尽，黄河入海流。欲穷千里目，更上一层楼。
#The primary use of LLaMA is research on large language models, including'''
#print("Test text:\n",text)
#print(f"Tokenized by LLaMA tokenizer:{llama_tokenizer.tokenize(text)}")
#print(f"Tokenized by Chinese-LLaMA tokenizer:{tr_llama_tokenizer.tokenize(text)}")