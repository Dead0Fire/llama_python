import torch
import struct

def read_floats(file,count):
    return struct.unpack(str(count)+'f',file.read(4*count))

def checkpoint_init_weights(path):
    with open(path,'rb') as f:
        dim,hidden_dim,n_layers,n_q_heads,n_kv_heads,vocab_size,seq_len=struct.unpack('7i',f.read(4*7))
        shared_weights=1 if vocab_size>0 else 0
        vocab_size=abs(vocab_size)
        head_size=dim//n_q_heads
        kv_dim=head_size*n_kv_heads

        token_embedding_table = read_floats(f, vocab_size * dim)
        rms_att_weight = read_floats(f, n_layers * dim)
        wq = read_floats(f, n_layers * dim * dim)
        wk = read_floats(f, n_layers * dim * kv_dim)
        wv = read_floats(f, n_layers * dim * kv_dim)
        wo = read_floats(f, n_layers * dim * dim)
        rms_ffn_weight = read_floats(f, n_layers * dim)
        w1 = read_floats(f, n_layers * dim * hidden_dim)
        w2 = read_floats(f, n_layers * hidden_dim * dim)
        w3 = read_floats(f, n_layers * dim * hidden_dim)
        rms_final_weight = read_floats(f, dim)
        freq_cis_real = read_floats(f, seq_len * head_size // 2)
        freq_cis_imag = read_floats(f, seq_len * head_size // 2)
        wcls = token_embedding_table if shared_weights else read_floats(f, vocab_size * dim)

    embedding_table = torch.tensor(token_embedding_table).view(vocab_size, dim)
    rms_final_w = torch.tensor(rms_final_weight)
    wcls_t = torch.tensor(wcls).view(vocab_size, dim).T

    layers = []
    for l in range(n_layers):
        layer = {}
        layer['n_q_heads'] = n_q_heads
        layer['qkv_head_size'] = head_size
        layer['n_kv_heads'] = n_kv_heads
        layer['Wq'] = torch.tensor(wq[l*dim*dim:(l+1)*dim*dim]).view(dim, dim).T
        layer['Wk'] = torch.tensor(wk[l*dim*dim:(l+1)*dim*kv_dim]).view(kv_dim, dim).T
        layer['Wv'] = torch.tensor(wv[l*dim*dim:(l+1)*dim*kv_dim]).view(kv_dim, dim).T
        layer['Wo'] = torch.tensor(wo[l*dim*dim:(l+1)*dim*dim]).view(dim, dim).T
        layer['W_rms'] = torch.tensor(rms_att_weight[l*dim:(l+1)*dim])
        layer['W_ffn'] = torch.tensor(rms_ffn_weight[l*dim:(l+1)*dim])
        layer['W1'] = torch.tensor(w1[l*dim*hidden_dim:(l+1)*dim*hidden_dim]).view(hidden_dim, dim).T
        layer['W2'] = torch.tensor(w2[l*hidden_dim*dim:(l+1)*hidden_dim*dim]).view(dim, hidden_dim).T
        layer['W3'] = torch.tensor(w3[l*dim*hidden_dim:(l+1)*dim*hidden_dim]).view(hidden_dim, dim).T
        layer['k_cache'] = torch.zeros(seq_len, n_kv_heads, head_size)
        layer['v_cache'] = torch.zeros(seq_len, n_kv_heads, head_size)
        layers.append(layer)

    return embedding_table, layers, rms_final_w, wcls_t
def rmsnorm(x,weight,eps=1e-8):
    return weight*x/torch.sqrt(torch.mean(x**2,dim=-1,keepdim=True)+eps)

def softmax(x,dim=-1):
    return torch.exp(x-torch.max(x,dim=dim,keepdim=True).values)/torch.sum(torch.exp(x-torch.max(x,dim=dim,keepdim=True).values),dim=dim,keepdim=True)

def ffn(x,w1,w2,w3):
    return (silu(x @ w1) * (x @ w3)) @ w2

def silu(x):
    return x * torch.sigmoid(x)

def attention(x,Wq,Wk,Wv,pos,n_q_heads,qkv_head_size,n_kv_heads,k_cache,v_cache):
    Q=x@Wq
    K=x@Wk
    V=x@Wv
    Q=Q.view(n_q_heads,qkv_head_size)
    K=K.view(n_kv_heads,qkv_head_size)
    V=V.view(n_kv_heads,qkv_head_size)
    Q=rope(Q,pos)
    K=rope(K,pos)
    k_cache[pos]=K
    v_cache[pos]=V
    group_size=n_q_heads//n_kv_heads #比例
    outs=[]
    for h in range(n_q_heads):
        kv_h=h//group_size  #分到哪个组
        q=Q[h]
        k=k_cache[:pos+1,kv_h,:]
        v=v_cache[:pos+1,kv_h,:]
        score=q@k.transpose(-2,-1)/torch.sqrt(torch.tensor(qkv_head_size,dtype=torch.float32))
        score=softmax(score)
        out=score@v
        outs.append(out)
    return torch.cat(outs,dim=-1)

def transformer_layer(x, Wq, Wk, Wv, Wo,W_rms,W_ffn,W1, W2, W3, pos, n_q_heads, qkv_head_size, n_kv_heads, k_cache, v_cache):
    h=rmsnorm(x,W_rms)
    att=attention(h,Wq,Wk,Wv,pos,n_q_heads,qkv_head_size,n_kv_heads,k_cache,v_cache)
    att=att@Wo
    x+=att
    h=rmsnorm(x,W_ffn)
    out=ffn(h,W1,W2,W3)
    x+=out
    return x

def rope(Q_K,pos):
    headsize=Q_K.shape[-1]
    fre=Q_K.new_tensor([10000**(-i/headsize) for i in range(0,headsize,2)])
    val=pos*fre
    fcr = torch.cos(val)
    fci = torch.sin(val)
    even = Q_K[..., 0::2].clone()
    odd = Q_K[..., 1::2].clone()

    Q_K[..., 0::2] = even * fcr - odd * fci
    Q_K[..., 1::2] = odd * fcr + even * fci

    return Q_K

def forward(token,pos,embedding_table,layers,rms_final_w,wcls):
    x=embedding_table[token].clone()
    for layer in layers:
        x=transformer_layer(x,layer['Wq'],layer['Wk'],layer['Wv'],layer['Wo'],layer['W_rms'],layer['W_ffn'],layer['W1'],layer['W2'],layer['W3'],pos,layer['n_q_heads'],layer['qkv_head_size'],layer['n_kv_heads'],layer['k_cache'],layer['v_cache'])
    x=rmsnorm(x,rms_final_w)
    logits=x@wcls
    return logits

def top_p(logits,p,temperature):
    sorted_logits,sorted_idx =torch.sort(logits,descending=True)
    sorted_logits=sorted_logits/temperature
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    cumulative_mask = cumulative_probs > p
    cumulative_mask[..., 1:] = cumulative_mask[..., :-1].clone()
    cumulative_mask[..., 0] = 0
    sorted_logits[cumulative_mask] = float('-inf')
    prob = torch.softmax(sorted_logits, dim=-1)
    out = torch.empty_like(prob)
    out[sorted_idx] = prob
    return out
def sample(logits,temperature,top_p_value):
    if temperature <= 0:
        return torch.argmax(logits).item()
    prob = top_p(logits,top_p_value,temperature)
    return torch.multinomial(prob, num_samples=1).item() #multinomial根据概率分布采样
#温度大于0 就根据概率分布采样，温度越高，采样越随机，温度越低，采样越确定，这个除以温度是为了控制采样的随机性，温度越高，softmax的输出分布越平坦，采样越随机；温度越低，softmax的输出分布越尖锐，采样越确定。

import re

def _unescape_token(s):
    # 把词表里字面的 <0xXX> 转义还原成真正的字节
    def repl(m):
        return bytes([int(m.group(1), 16)])
    return re.sub(r'<0x([0-9A-Fa-f]{2})>', lambda m: repl(m).decode('latin1'), s)

def tokenizer_init(path,vocab_size):
    vocab,vocab_scores=[],[]
    with open(path,'rb') as f:
        max_token_length=struct.unpack('i',f.read(4))[0]
        for _ in range(vocab_size):
            vocab_scores.append(struct.unpack('f',f.read(4))[0])
            length=struct.unpack('i',f.read(4))[0]
            raw = f.read(length)
            # 先按 utf8 解码；失败则按 latin1 保留原始字节
            try:
                token = raw.decode('utf8')
            except UnicodeDecodeError:
                token = raw.decode('latin1')
            vocab.append(_unescape_token(token))
    return vocab,vocab_scores,max_token_length
#词表读取

def str_lookup(token,vocab):
    try:
        return vocab.index(token)
    except ValueError:
        return -1
#vocab是一个列表，里面存储了所有的词汇，str_lookup函数的作用是查找一个字符串在vocab中的索引，如果找不到就返回-1。
def bpe_encode(text,vocab,vocab_scores):
    tokens=[]
    for pos,char in enumerate(text):# emuerate是一个内置函数，它可以同时获取元素的索引和值，这里用来遍历文本中的每个字符
        id =str_lookup(char,vocab)
        if id==-1:
            print(f"Warning: character '{char}' not found in vocabulary. Skipping.")
            raise SystemExit(1)
        tokens.append(id)
    while True:
        best_score=-float('inf')
        best_id=-1
        best_idx=-1 #这里的best是用来记录当前最优的子词分割方案的，best_score是当前最优的分数，best_id是当前最优的子词的索引，best_idx是当前最优的子词在文本中的位置。
        for i in range(len(tokens)-1):
            string = vocab[tokens[i]] + vocab[tokens[i + 1]]
            id=str_lookup(string,vocab)
            if id!=-1:
                if vocab_scores[id]>best_score:
                    best_score=vocab_scores[id]
                    best_id=id
                    best_idx=i
        if best_id==-1:
            break
        tokens[best_idx]=best_id
        del tokens[best_idx+1]
    return tokens

def generate(checkpoint,tokenizer_path,prompt,steps,temperature,top_p_value):
    embedding_table,layers,rms_final_w,wcls=checkpoint_init_weights(checkpoint)
    vocab_size=embedding_table.shape[0]
    seq_len=layers[0]['k_cache'].shape[0]
    vocab,vocab_scores,_=tokenizer_init(tokenizer_path,vocab_size)
    if steps<=0 or steps>seq_len:
        steps=seq_len
    prompt_tokens=bpe_encode(prompt,vocab,vocab_scores) if prompt else []
    token=1  # BOS
    pos=0
    print("<s>",end="")
    while pos<steps:
        logits=forward(token,pos,embedding_table,layers,rms_final_w,wcls)
        if pos<len(prompt_tokens):
            next_token=prompt_tokens[pos]
        else:
            next_token=sample(logits,temperature,top_p_value)
        token_str=vocab[next_token].lstrip() if token==1 and vocab[next_token][:1]==' ' else vocab[next_token]
        print(token_str,end="",flush=True)
        if next_token==1:  # EOS
            break
        token=next_token
        pos+=1


def main():
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    generate(
        checkpoint=os.path.join(base_dir, 'stories15M.bin'),
        tokenizer_path=os.path.join(base_dir, 'tokenizer.bin'),
        prompt="Once upon a time",
        steps=256,
        temperature=0.8,
        top_p_value=0.9,)
if __name__=="__main__":
    main()