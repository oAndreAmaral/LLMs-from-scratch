# This code implements the architecture of an LLM that is a bit more complex than our basic one
# This is needed for the alignment phase because to perform alignment we need a pretrained model that produces
# better results than the ones that can be achieved with our small LLM.

# I am providing the file of a checkpoint trained with this LLM (138 million parameters vs 19 of our initial basic one)
# so that we can apply ORPO alignment on top of it.
# The pretrained 138 million parameter model has been pre-trained on the open source Fineweb-Edu dataset.

# With this extra code, you will be able to reinforce your understanding of the internals of LLMs. It has the very same parts than 
# our basic one, but at the same time it is structured a little bit differently and some of the parts are done in a 
# more sophisticated way. So it will be very educational and useful for you to compare and understand both codes

# This file includes code from the open source LLaMA2.c repository licensed under the MIT License.
# See licenses/llama-license for details.
# Modifications: Variable names have been changed for educational purposes.

# Official Notebook

# Import libraries

import os, re
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import ipdb
import argparse

# Pytorch
import torch
from torch import nn
import torch.nn.functional as F

# Architecture
import transformers

# automatically define methods for class
@dataclass
class ModelArgs:
    # Notice that these are default values for Llama, these are not the ones we use in our pretrained file
    dim: int = 4096 #  dimensionality of internal embeddings / hidden states
    n_layers: int = 32  # number of layers
    n_heads: int = 32  # number of query heads per transformer block
    n_kv_heads: int = 8  # number of key and value heads
    vocab_size: int = 128256  # vocab size
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None # specifies a multiplier for the dim of hidden layers in FFN stages
    norm_eps: float = 1e-5 # value used for numerical stability
    rope_theta: float = 500000 # controls the scaling factor used in the rotational positional encoding
    max_seq_len: int = 8192 # context size
    dropout: float = 0.1  # dropout percentage
    hidden_dim: int = 14336, # hidden internal larger dimension for feedforward network
    attention_bias: bool = True, # do we use bias in attention layers?
    mlp_bias: bool = True  # do we use bias in mlp layers?

# Our pretrained saved checkpoint has these parameters:
# ModelArgs(dim=768, n_layers=12, n_heads=12, n_kv_heads=12, vocab_size=16384, multiple_of=256,
# ffn_dim_multiplier=None, norm_eps=1e-06, rope_theta=10000.0, max_seq_len=1024, dropout=0.0, hidden_dim=3072,
# attention_bias=False, mlp_bias=False)


# RMS Normalization
# RMSNorm normalizes the inputs based on the root mean square (RMS) of the input values, 
# without centering them (i.e., without subtracting the mean).
# It is computationally efficient and can lead to stable gradients but offers slightly less flexibility

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) # learnable scaling factor for the norm

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # it divides x by the square root of the mean squared value of the input

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# Compute frequency vectors used in the context of rotary position encodings (RoPE) 
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  
    freqs = torch.outer(t, freqs).float()  
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    # We get frequency vectors (freqs_cos and freqs_sin) used in rotary positional encoding mechanisms within LLMs
    # Frequencies are computed based on a geometric progression to ensure different positions receive distinct encoding patterns.

    return freqs_cos, freqs_sin

# Helper function for Rotary position encodings (RoPE) 
# Ensures that freqs_cis tensor can broadcast correctly across x by reshaping it to match the dimensions required for broadcasting operations.
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

# Rotary position encodings (RoPE) 
# Applies rotary embeddings to input tensors xq and xk using cosine and sine components
# (freqs_cos and freqs_sin), reshaped and prepared for broadcasting, 
# resulting in transformed outputs xq_out and xk_out

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


# Matching heads
# Expand or replicate the key (k) and value (v) heads to match the number of query (q) heads.
# (not needed in our case because we use same number of q, k and v heads)
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

# Attention Mechanism Class
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads # we keep all numbers of heads the same
        assert args.n_heads % self.n_kv_heads == 0
        
        model_parallel_size = 1
        # degree of model parallelism being used (splitting a model across multiple devices or processors to handle
        # larger models that do not fit into a single GPU or memory space. (we set it to 1 so it has no impact at all)

        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size

        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 1 , no replication needed

        self.head_dim = args.dim // args.n_heads  # 768 // 12 = 64 in our case
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=args.attention_bias) # (768, 768)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias) # (768, 768)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias) # (768, 768)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=args.attention_bias) # (768, 768)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # use flash attention or a manual implementation?
        # Flash Attention is a highly efficient implementation of the scaled dot-product attention mechanism.
        # It aims to accelerate the attention computation while maintaining accuracy. (requires Pytorch >= 2.0)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') # detect flash attention
        self.flash = False 
        # at the time of creating this notebook, there is a bug in a library that crashes when using flash attention
        # so set self.flash to False if the bug happens or if your system is not compatible with flash attention
        # Set it also to false if you prefer to run or debug the flash computations manually

        if not self.flash:
            # Prepare a mask to perform manual attention -> triangular mask where the bottom left triangle + diagonal are 
            # set to 0, and the top right triangle to is set to -infinity
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf")) # (1,1,1024,1024) all set to -inf
            mask = torch.triu(mask, diagonal=1) # bottom left triangle + diagonal set to 0, rest is -inf

            #self.register_buffer("mask", mask) # we cannot use buffer because the saved pretrained doesn't have this variable
            self.mask = mask
            self.mask = self.mask.to(device) 
            

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ):
        # BS: Batch Size / SL: Sequence length or context
        bsz, seqlen, _ = x.shape

        # input x shape = (BS, SL, Embedding size) -> (BS,1024,768)
        # head_dim = Embedding dim // n_ḧeads = 768 // 12 = 64
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)  # (BS, SL, Embedding size) -> (BS,1024,768)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim) # (BS, SL, n_local_heads, head_dim) -> (BS,1024,12,64)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim) # (BS,1024,12,64)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim) # (BS,1024,12,64)

        # RoPE relative positional embeddings
        # in this architecture, the rotary embeddings are applied on every layer to q and k, so we also receive them here
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin) # (BS, SL, n_local_heads, head_dim) -> (BS,1024,12,64)

        # grouped multiquery attention: expand out keys and values
        # this makes no effect in our case, no expansion needed as we have same number of q, k and v heads
        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)  # (BS,1024,12,64)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)  # (BS,1024,12,64)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim) # (BS,12,1024,64)
        xk = xk.transpose(1, 2)  # (bs,12,1024,64)
        xv = xv.transpose(1, 2)  # (bs,12,1024,64)

        # flash implementation: if active the whole thing will be done automatically by a single function
        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # manual implementation
            # Produce matrix of attention weights by multiplying Q and K, scaled by sqrt of head_dim
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim) # (BS,12,1024,1024)

            # mask = (1,1,1024,1024) # -inf in top right triangle
            # apply mask to hide future tokens 
            scores = scores + self.mask[:, :, :seqlen, :seqlen]   # (BS, n_local_heads, SL, SL) (BS,12,1024,1024)

            # Convert to probabilities
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)  # (BS,12,1024,1024) scores[0][0][0].sum() = 1
            scores = self.attn_dropout(scores)  # apply dropout

            # Apply attention weights to V
            output = torch.matmul(scores, xv)  # (BS, n_local_heads, SL, head_dim) # (BS,12,1024,64)

        
        # Restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)  # (BS,1024,768)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output


# FeedForward Network
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float, bias: bool = True):
        super().__init__()
        if hidden_dim is None: # it is not None in our case
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias) #(768, 3072)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias) #(3072, 768)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias) #(768, 3072)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # BS: Batch Size / SL: Sequence length or context
        # x -> (BS, SL, 768)
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x))) # output -> (BS, SL, 768)


# Transformer Blocks
class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads # 12 in our case
        self.dim = args.dim         # 768 in our case
        self.head_dim = args.dim // args.n_heads  # 64 in our case
        self.attention = Attention(args) 
        self.feed_forward = FeedForward(  
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
            bias=args.mlp_bias
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        # BS: Batch Size / SL: Sequence length or context
        # x -> (BS, SL, 768)
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin) # (BS, SL, 768)
        out = h + self.feed_forward.forward(self.ffn_norm(h)) # (BS, SL, 768)
        return out


#################################################################################
################## LLM LLAMA BASED MODEL ########################################
# 138 Million Parameters in the default configuration
###############################################        
##################################

class Llama(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size # 16384 in our case
        self.n_layers = params.n_layers  # 12 in our case

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim) # Setup embedding layers
        self.dropout = nn.Dropout(params.dropout) # Setup dropout

        # Create a list of modules, and append to it a number of Transformer Blocks (12 layers)
        self.layers = torch.nn.ModuleList() 
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # some useful precompute for the RoPE relative positional embeddings
        freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len, theta=self.params.rope_theta)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # Initialize attribute for the loss of the last forward call.
        # This will be set if the forward is called with a targets tensor.
        self.last_loss = None

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        # BS: Batch Size / SL: Sequence length or context

        BS, SL = tokens.shape  # (BS, SL) # SL is 1024 in our case
        h = self.tok_embeddings(tokens)  # (BS, SL, 768)
        h = self.dropout(h) # (BS, SL, 768)
        freqs_cos = self.freqs_cos[:SL]  # (SL,32)
        freqs_sin = self.freqs_sin[:SL]  # (SL,32)

        # Go through the different Transformer Blocks
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin) # (BS, SL, 768)
        h = self.norm(h)  #(BS,SL,768)

        # Calculate Loss if training with targets

        # Cross Entropy Logic
        # (equivalent to negative log likelihood)

        # Information: -log p(x) (inverse of probability)
        # Entropy: avg of information in random variable (prob distribution): - sum_x (x * log(x))
        # CrossEntropy: Compares 2 distr q(true) & p(predicted) in terms of information distance: -sum_x (q(x) * log p(x))
        # LLMs CrossEntropy: true labels are 1 for true, 0 for the rest, so it simplifies to: -sum_x log p(x)
                
        if targets is not None:
            # if we are given some desired targets also calculate the loss

            # The logits are the predictions of the network (not yet as probabilities)
            logits = self.output(h) # (BS,SL,16384) 16384 is the vocab size

            # Shift the labels one to the left to pair each input with the next token in the label
            shift_logits = logits[..., :-1, :].contiguous() # (BS,1023,16384)  1023 is SL-1
            shift_labels = targets[..., 1:].contiguous()  # (BS,1023)
            
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.params.vocab_size) # Reshape for cross_entropy (BS*1023, 16384)
            shift_labels = shift_labels.view(-1) # Reshape as well to prepare (BS*1023)

            self.last_loss = F.cross_entropy(shift_logits, shift_labels)

            # Optional: Just for fun, manual way to calculate cross_entropy
            # By default, we comment out the manual version to prevent calculating the loss twice (will make things slower)
            
            # First apply softmax to produce probabilities
            #counts = shift_logits.exp()  # (BS*1023,16384)
            #prob = counts / counts.sum(-1, keepdim=True) # (BS*1023,16384) / (BS*1023,1) = (BS*1023,16384)

            # We need to ignore all the labels that are -100 for the loss calculation
            #mask = (shift_labels != -100)
            #filtered_prob = prob[mask]
            #filtered_labels = shift_labels[mask]

            #log_prob = torch.log(filtered_prob[torch.arange(filtered_labels.size(0)), filtered_labels])
            #self.last_loss2 = -log_prob.mean()

            # Finally at each of prob's positions, we pick the index specified by the respective target
            # example shift_labels[3]=329, prob[3][329] = 0.014

            # Most times they will match, sometimes they will not because F.cross_entropy is more precise
            # Uncomment if you want to see when both loss calculations don't match
            #if ( not torch.allclose(self.last_loss,self.last_loss2)):
            #    print(f"[Loss Diff] Pytorch:{self.last_loss.item()} Manual:{self.last_loss2.item()}")

        else:
            # inference-time mini-optimization: only forward the output on the very last position
            logits = self.output(h[:, [-1], :]) # note: using list [-1] to preserve the time dim
            self.last_loss = None

        return logits, self.last_loss


    #####################################################
    ############ GENERATE NEW SAMPLE ####################
    #####################################################

    # We take a conditioning sequence of indices idx (BS,SL) and complete
    # the sequence max_new_tokens times, feeding the predictions back into the model each time.

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # BS: Batch Size / SL: Sequence length or context    
 
        for _ in range(max_new_tokens):

            # if the sequence context is growing too long we must crop it at a max of SL size
            idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]  #(1,SL)

            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)  #(1,1,16384)  #forward function of model returns just the probs of very last position
            logits = logits[:, -1, :] # crop the second dimension #(1,16384)            

            if temperature == 0.0:
                # "sample" the single most likely index
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # pluck the logits at the final step and scale by desired temperature
                logits = logits / temperature
                # Controlling Confidence and Randomness:
                # Low Temperature (< 1, 0.2, 0.4): When the temperature is low, the logits are scaled up, 
                # which makes the softmax function output probabilities that are more confident (closer to 0 or 1). 
                # This results in less random and more deterministic predictions.
                # High Temperature (> 1): When the temperature is high, the logits are scaled down, 
                # making the softmax output probabilities more evenly distributed. 
                # This introduces more randomness and exploration into the predictions.

                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)  # (1,16384)
                idx_next = torch.multinomial(probs, num_samples=1)  # returns single token value

            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
    ######### Clean responses
    def clean_response(self,response):
        #ipdb.set_trace()

        # Remove user and assistant tags
        response = response.replace("<|user|>", "")
        response = response.replace("<|assistant|>", "")
        
        # Remove anything within square brackets
        response = re.sub(r'\[.*?\]', '', response)

        # Optionally, remove extra spaces or newlines left behind
        #response = re.sub(r'\s+', ' ', response).strip()

        return response

#####################################################
############ RUNNING LLM.PY IN INFERENCE MODE########
#####################################################

if __name__ == "__main__":  

    parser = argparse.ArgumentParser(description='llm')
    parser.add_argument('-align',action='store_true',help='Enable alignment')
    parser.add_argument('-num',type=int, default=1, help='Number of answers')
    parser.add_argument('-temp',type=float, default=0.5, help='Temperature value')
    parser.add_argument('-topk',type=int, default=50, help='Topk value')

    args=parser.parse_args()
    use_orpo = args.align  # use aligned checkpoint or not
    num_answers = args.num
    temp = args.temp
    topk=args.topk

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer_path = "tokenizers/tok16384"
    model_path = "./models/"
    
    if use_orpo==True:
        model_inf, context= "aligned_model.pt", 1024  # ORPO is trained with context of 1024
        print("Mode::Using Orpo aligned model")
    else:
        model_inf, context= "base_model.pt", 512  # The original was trained with context of 512
        print("Mode::Using pretrained model without alignment")

    print(f"Using model {model_inf}")
   
    # Load model and extract config
    checkpoint = torch.load(os.path.join(model_path, model_inf), map_location=device, weights_only=False)
    config = checkpoint.pop("config")
    
    # temporary fix if the model was trained and saved with torch.compile
    # The _orig_mod. prefix in your model's state dictionary keys is related to
    # how PyTorch handles compiled models, specifically when using the torch.compile function
    # When torch.compile is used, PyTorch might wrap the original model in a way that modifies
    # the names of its parameters and buffers. This wrapping can prepend a prefix like _orig_mod.
    # We remove those wrappings to make the checkpoint compatible with the non compiled version of the model
    new_dict = dict()
    for k in checkpoint.keys():
        if k.startswith("_orig_mod."):
            #print("Removing _orig_mod wrapping")
            new_dict[k.replace("_orig_mod.", "")] = checkpoint[k]
        else:
            new_dict[k] = checkpoint[k]

    # Setup tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    model_args = ModelArgs(
        dim=config.hidden_size, 
        n_layers=config.num_hidden_layers, 
        n_heads=config.num_attention_heads, 
        n_kv_heads=config.num_key_value_heads, 
        vocab_size=config.vocab_size, 
        norm_eps=config.rms_norm_eps, 
        rope_theta=config.rope_theta,
        max_seq_len=context, 
        dropout=config.attention_dropout, 
        hidden_dim=config.intermediate_size,
        attention_bias=config.attention_bias,
        mlp_bias=config.mlp_bias
    )

    # Instantiate model, load parms, move to device
    model = Llama(model_args)
    model.load_state_dict(new_dict)
    if device.type == 'cuda':
        model = model.to(torch.bfloat16)
        model = model.to(device)
    model.eval()

    model_size = sum(t.numel() for t in model.parameters())
    print(f"Model size: {model_size/1e6:.2f} M parameters")

    # Interactive loop
    while True:
         qs = input("Enter text (q to quit) >>> ")
         if qs == "":
             continue
         if qs == 'q':
             break
  
         # we activate chat template only for ORPO model because it was trained with it
         if use_orpo:
            qs = f"<s> <|user|>\n{qs}</s>\n<s> <|assistant|> "

         x = tokenizer.encode(qs)
         x = torch.tensor(x, dtype=torch.long, device=device)[None, ...]

         for ans in range(num_answers):
            with torch.no_grad():
                y = model.generate(
                    x, 
                    max_new_tokens=256, 
                    temperature=temp, 
                    top_k=topk
                )

            response = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)   

            output = model.clean_response(response)

            print("################## \n")
            print(f"### Answer {ans+1}: {output}")

        

