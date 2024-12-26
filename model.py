import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

class TokenEmbedding(nn.Module):
    """
    Class that goes from raw tokens to embedding dimension.
    """
    def __init__(self, vocab_size, emb_dim):
        super(TokenEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.token_embedding = nn.Embedding(self.vocab_size, self.emb_dim)
        
    def forward(self, x):
        # print(f"inside tokenembedding: {x.shape}")
        out = self.token_embedding(x)
        # print(f"after token embedding: {out}")
        # print(f"after token embedding shape: {out.shape}")
        return out
    
class PositionalEmbedding(nn.Module):
    """
    Class that provides embedding that captures the positional information of a token.
    """
    def __init__(self, block_size, emb_dim):
        super(PositionalEmbedding, self).__init__()
        self.block_size = block_size
        self.emb_dim = emb_dim
        # Use torch.no_grad to make sure PositionalEmbedding not part of computational graph
        with torch.no_grad():
            # Credit to https://github.com/hyunwoongko/transformer 
            self.pos_embedding = torch.zeros(self.block_size, self.emb_dim)
            position = torch.arange(self.block_size).unsqueeze(dim=1)
            _2i = torch.arange(0, self.emb_dim, step=2)
            self.pos_embedding[:, 0::2] = torch.sin(position / (10000 ** (_2i / self.emb_dim)))
            self.pos_embedding[:, 1::2] = torch.cos(position / (10000 ** (_2i / self.emb_dim)))
    
    def forward(self, x):
        seq_len = x.size(1)
        # print(f"After pos_embedding: {pos_embedding}")
        # print(f"pos_embedding shape: {pos_embedding.shape}")
        return self.pos_embedding[:seq_len, :]

class MultiHeadAttention(nn.Module):

    """
    Class that performs the splitting of Q, K, V into separate attention calculations, concatenating at the end.
    """

    def __init__(self, block_size, emb_dim, model_dim, num_heads, mask=False):
        super(MultiHeadAttention, self).__init__()
        self.block_size = block_size
        self.emb_dim = emb_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.mask = mask
        self.Q = nn.Linear(model_dim, model_dim) #  Model dim == embed_dim, so this works for both the first encoder and all following encoders
        self.K = nn.Linear(model_dim, model_dim) 
        self.V = nn.Linear(model_dim, model_dim)
        self.o = nn.Linear(model_dim, model_dim)
        self.attention = ScaledDotProductionAttention(mask)

    def forward(self, q, k, v):
        # print(f"Input before attention{q.shape}")
        q, k, v = self.split(self.Q(q)), self.split(self.K(k)), self.split(self.V(v)) # queries, keys, values
        output = self.attention(q, k, v, self.model_dim)
        output = self.concatenate(output)
        # print(f"Shape of attention after concatenate and before w_o: {output.shape}")
        output = self.o(output)
        # Need to pass through w_o here, which should be (h*c_heads, model_dim)

        return output

    # Go from (B, T, C) --> (B, h, T, C // h). There's other ways of doing this, but people have found this is efficient
    def split(self, x):
        # print(f"pre-split {x.shape}")
        B, T, C = x.shape
        out = x.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # Intuitively, each channel's/head's calculations are separate from each other, just like different channels in a CNN
        return out
    
    def concatenate(self, x):
        B, n_heads, T, C_per_head = x.shape
        # print(f"Before concatenate: {x.shape}") # It's (B, n_heads, T, C_per_head)
        out = x.transpose(1,2).contiguous().view(B, T, n_heads * C_per_head) # Want to be (B, T, T*n_heads)
        # print(f"After concatenate: {out.shape}")
        return out
    
## I see, so don't necessarily need to have the q, k, v defined in here, this can just serve as a function
class ScaledDotProductionAttention(nn.Module): 
    """
    Class that performs the attention calculation using Q, K, V.
    """
    def __init__(self, mask):
        super(ScaledDotProductionAttention, self).__init__()
        self.mask = mask
        self.softmax = nn.Softmax(dim=-1) # Want to be along rows
    
    # Assume q, k, v are the tensors, already multiplied by the Q, K, V weights. They should also be of shape (B, T, C/h, h)
    def forward(self, q, k, v, model_dim):
        # Checking shapes of q, k, v
        # print(f"q: {q.shape}")
        # print(f"k: {k.shape}")
        # print(f"v: {v.shape}")
        pre_softmax = (q @ k.transpose(-1, -2)) / (model_dim ** 0.5) 
        if self.mask:
            m = torch.ones_like(pre_softmax)
            m = torch.tril(m)
            pre_softmax = torch.where(m != 0, pre_softmax, float('-inf'))
        

        # print(f"Before softmax shape: {pre_softmax.shape}")
        post_softmax = self.softmax(pre_softmax) # Mask gets applied before the softmax so negative infinities get set to 0 during softmax
        # print(f"After softmax shape: {post_softmax.shape}") # This is just a sanity check: softmax should not change shape of input
        out = post_softmax @ v
        # print(f"attention (softmax @ v): output shape after split before concatenate: {out.shape}")
        return out 

class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-forward Network made up of 2 Linear Layers applied to each position independently. 
    """
    def __init__(self, model_dim, hidden_dim):
        super(FeedForwardNetwork, self).__init__()
        self.linear_1 = nn.Linear(model_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, model_dim)

    def forward(self, x):
        out = F.relu(self.linear_1(x))
        out = self.linear_2(out)
        return out

class ProjectionHead(nn.Module):
    """
    Projection head for the Decoder to map back to probabilities for next token
    """
    def __init__(self, vocab_size, model_dim):
        super(ProjectionHead, self).__init__()
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.linear = nn.Linear(model_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        out = self.linear(x)
        out = self.softmax(out)
        return out

class EncoderBlock(nn.Module):
    """
    Encoder block made up of a self-attention, layer norm, residual connection, and FFN.
    """
    def __init__(self, block_size, emb_dim, model_dim, num_heads, mask):
        super(EncoderBlock, self).__init__()
        # don't need the embeddings here because they only get used once
        self.block_size = block_size
        self.emb_dim = emb_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask
        self.ln_1 = nn.LayerNorm(model_dim)
        self.ln_2 = nn.LayerNorm(model_dim)

        self.mha = MultiHeadAttention(block_size, emb_dim, model_dim, num_heads, mask) # q, k, v all defined within here
        self.ffn = FeedForwardNetwork(model_dim, model_dim)
        
    
    def forward(self, x):
       
        x_norm = self.ln_1(x) # Using pre-norm method due to increased performance
        
        attention = self.mha(q=x_norm, k=x_norm, v=x_norm)
        # print(attention.shape)
        x = x + attention # first residual (the norm is inside the residual)
        
        attention_norm = self.ln_2(x)
        # print(f"attention shape before FFN {attention_norm.shape}")
        output = self.ffn(attention_norm)
        output = output + x 

        return output 


class Encoder(nn.Module):
    """
    Encoder class that makes up the Encoder side of the network
    """
    def __init__(self, vocab_size, block_size, num_blocks, emb_dim, model_dim, num_heads):
        super(Encoder, self).__init__()
        self.pos_emb = PositionalEmbedding(block_size, emb_dim)
        self.tok_emb = TokenEmbedding(vocab_size, emb_dim)
        self.encoders = nn.ModuleList([EncoderBlock(block_size, emb_dim, model_dim, num_heads, mask=False) for i in range(num_blocks)])
        
    def forward(self, x):
        x = self.tok_emb(x) + self.pos_emb(x) # Applying the token and positional embeddings to get input for first encoder block
        # print(f"Shape after embedding: {x.shape}")
        # print(f"x after tok + pos: {x}")
        for i, l in enumerate(self.encoders):
            # print(f"On this encoder block: {i}")
            x = l(x)
            # print(f"{i}'th encoder, x: {x}")
        return x 

class DecoderBlock(nn.Module):
    """
    Class defining the decoder block.
    """
    def __init__(self, block_size, emb_dim, model_dim, num_heads, mask):
        super(DecoderBlock, self).__init__()
        self.block_size = block_size
        self.emb_dim = emb_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask
        self.ln_1 = nn.LayerNorm(model_dim)
        self.ln_2 = nn.LayerNorm(model_dim)
        self.ln_3 = nn.LayerNorm(model_dim)

        self.self_attention = MultiHeadAttention(block_size, emb_dim, model_dim, num_heads, mask) # q, k, v all defined within here
        self.cross_attention = MultiHeadAttention(block_size, emb_dim, model_dim, num_heads, mask) 
        self.ffn = FeedForwardNetwork(model_dim, model_dim)
    
    # dec, enc are the output of the previous decoder and the final encoder respectively
    def forward(self, dec, enc=None):
        # ln1_dim = dec.shape[-1]
        # ln_1 = nn.LayerNorm(ln1_dim)
        pre_x = self.ln_1(dec) # Using pre-norm method due to increased performance
        # print(dec.shape)
        x = self.self_attention(q=pre_x, k=pre_x, v=pre_x)
        x = x + dec # First residual connection in the block
        # print(x.shape)
        # ln2_dim = x.shape[-1]
        # ln_2 = nn.LayerNorm(ln2_dim)
        x_norm = self.ln_2(x)
        # print(f"attention shape after self attention {x.shape}")
        if enc is not None:
            cross_attention = self.cross_attention(q=x_norm, k=enc, v=enc)
            x = x + cross_attention # second residual connection applied after cross attention

        x_norm = self.ln_3(x) 
        out = self.ffn(x_norm)
        out = out + x
        
        return out
    
class Decoder(nn.Module):
    """
    Class defining the decoder.
    """
    def __init__(self, vocab_size, block_size, num_dec_blocks, emb_dim, model_dim, num_heads):
        super(Decoder, self).__init__()
        self.pos_emb = PositionalEmbedding(block_size, emb_dim)
        self.tok_emb = TokenEmbedding(vocab_size, emb_dim)
        self.decoders = nn.ModuleList([DecoderBlock(block_size, emb_dim, model_dim, num_heads, mask=True) for i in range(num_dec_blocks)])
        self.proj_head = ProjectionHead(vocab_size, model_dim)

    def forward(self, x):
        x = self.tok_emb(x).to(x.device) + self.pos_emb(x).to(x.device) # Applying the token and positional embeddings to get input for first encoder block
        # print(f"Shape after embedding: {x.shape}")
        # print(f"x after tok + pos: {x}")
        for i, l in enumerate(self.decoders):
            # print(f"On this decoder block: {i}")
            x = l(x)
            # print(f"{i}'th decoder, x: {x}")
        probs = self.proj_head(x)
        # print(f"final output probabilities: {x}")
        return probs 

if __name__ == '__main__':
    # Let's imagine the vocab_size is 20

    input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]]) # These are the tokens, so dimension of (batch, time)
    B, T = input.shape
    # print(f"input shape {input.shape}")
    decoder = Decoder(vocab_size=20, block_size=T, num_dec_blocks=6, emb_dim=8, model_dim=8, num_heads=2)
    output = decoder(input)
    # print(f"final output: {output}")
    # print(f"output shape:{output.shape}")

    # encoder = Encoder(vocab_size=20, block_size=T, num_blocks=6, emb_dim=8, model_dim=8, num_heads=2)
    # output = encoder(input)
    # print(f"final output: {output}")
    # print(f"output shape:{output.shape}")


    


        




