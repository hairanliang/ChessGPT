
# ChessGPT

I built and trained a GPT model that can generate chess games in PGN notation from scratch in PyTorch. I attached it to an API using FastAPI and containerized it using Docker.



## Overview

Generative pre-trained transformer, or GPT, models have taken the world by storm. ChatGPT and similar LLMs are being used by students, parents, and enterprises across the world to solve a variety of problems. While I knew the basic theory of transformers, I wanted to delve deeper and actually create one from scratch by following the original 
[AttentionIsAllYouNeed](https://arxiv.org/abs/1706.03762) paper. 

Having played professional chess for close to half of my life, I wanted to tie the project to chess somehow, which is how I came up with the idea of Chess-GPT. Many of us may be familiar with the famous case of AlphaZero, a reinforcement learning computer that dominated the chess scene. It not only beat the best human players, but more importantly, the strongest chess engines. My goal was not to dethrone AlphaZero——my hunch is that reinforcement learning based models will always be the king of chess engines——but to create a GPT model that could produce reasonable chess games.

With this project, I wanted to learn how to code up the Transformer architecture, and consequently the GPT-2 architecture all in PyTorch. I wanted to learn more about training enormous models by firsthand training one——my base model ended up with 75,222,251 parameters! Lastly, I wanted some more practical experience deploying deep learning models, so I experimented with FastAPI and Docker to containerize and wrap my model with an API that others can easily call and use to generate chess games, whether on a virtual machine or local machine.

## How do we convert chess games to text?

In order to build a GPT-model, we need some way to represent chess in text which can further be tokenized. Luckily for me, chess games have an easy way to be converted to text as it is ingrained into professional chess itself——chess PGN notation. In formal games, chess players are responsible for writing down each move made in the game. This serves two purposes: 1. If there is a disagreement later in the game and a referee needs to be involved, the players can refer to the game notation 2. Players can replay their games and learn from their mistakes or admire their victories (the former is less fun but infinitely more helpful for improvement...).

Here is an example of chess notation for the famous 4 move checkmate: 

1\. e4, e5 2. Bc4, Nc6 3. Qh5, Nf6 4. Qxf7# 

![mate_in_4](https://github.com/user-attachments/assets/4a05c5ac-4c60-420e-a72e-301d5e192409)

From the notation, it hints at how we can tokenize these chess games for our transformer model. 

## Tokenizer and Dataloader

At the end of the day, a chess game is just a sequence of moves. Thus, I tokenized each game by representing each move by a token. Thus, for each dataset of chess games, the total number of tokens is just the number of distinct moves. This varied based on the dataset, but for example, the number of tokens for a smaller dataset I used (Black Knight Tango) was only around 3000, whereas for the larger dataset, it was about 9000 tokens.

It's also important to note that besides the tokens for each unique chess move in the dataset, we also need special tokens for training. These tokens are the 'BOS' (Beginning of Sequence), 'EOS' (End of Sequence), and 'PAD' (Padding) tokens. 'BOS' is self-explanatory: it will be added to the beginning of each tokenized chess game in the dataloader: 

    def __getitem__(self, idx):
        
        prepend = 'BOS'
        
        # Have to make sure don't mutate it with each access, so just make a copy
        game = self.game_list[idx].copy() 
        
        game = [prepend] + game

This way, during training, when the model sees the 'BOS' token, it knows to then predict the first move of the chess game.

'EOS' tokens are also straight forward——they will be added to the end of each game so that the model knows that the game is over, signalling no more tokens coming afterwards.

The 'PAD' special token is very important in terms of batching and training. Within each batch, there could be different length games, which would be a huge problem if we want efficient training. So, to make things easier, we can simply pad every game to a fixed amount, such as 80 tokens——about the average length of a chess game (40 moves for each side). 

So, for every game in our dataset, if it is shorter than the max game length, we pad it to the max game length and append the 'EOS' token. If it's longer than the max length, we shrink the game and then append the 'EOS' token. 

        if len(game) >= self.max_game_length:
            game = game[:self.max_game_length - 1]
            game.append('EOS')
        else:
            game.append('EOS')
            while len(game) < self.max_game_length:
                game.append('PAD')

We then have to tokenize the games, and provide a mask that the training function will use to ensure that 'PAD' tokens do not affect the loss calculations and consequently backpropagation of the model's weights.

        tokenized_game = self.tokenize(game, self.token_dict)

        PAD_token = len(self.token_dict) - 2 # PAD token is second last token
        EOS_token = len(self.token_dict) - 1 # EOS token is last token
        loss_mask = torch.tensor([0 if token == PAD_token else 1 for token in tokenized_game])

        loss_mask = loss_mask[:-1] # This is to correct for the fact that we grabbed max_length + 1 tokens, since we are using the produce_pairs function

        return tokenized_game, loss_mask

Feel free to check out the dataloader.py file to see the actual tokenize function defined, and other details left out here.

Side note: In the future, it is also possible to incorporate dynamic padding, where instead of padding to a max game length for all games, we can do it batch-wise, padding every game to the longest length in that particular batch. This could speed up training because we aren't always padding games to a fixed number, and also allow for cases where we want to learn from games that are longer than 80 tokens for example.


## Datasets

Until now, I have two datasets that the model has been trained on. The first one is small dataset of 3915 games, all in one particular chess opening, the [Black Knight Tango](https://www.pgnmentor.com/files.html#openings). The purpose of this dataset was to see if the model could overfit to the first several moves, since all the games in this dataset share the first 4 chess moves, that is, 2 moves from white and 2 moves from black: 1.d4 Nf6 2.c4 Nc6. Fortunately, the model successfully produced games that overfit to the opening moves——here is one amusing example: 

['BOS', 'd4', 'Nf6', 'c4', 'Nc6', 'Nf3', 'd6', 'Nc3', 'e5', 'd5', 'Ne7', 'e4', 'g6', 'Be2', 'Be7', 'O-O', 'O-O', 'a6', 'Qc2', 'Bf5', 'f4', 'c5', 'g3', 'Qf6', 'cxd6', 'cxd6', 'cxd6', 'cxd6', 'cxd6', 'cxd6', 'cxd6', 'cxd6', 'g3', 'd5', 'exd5', 'cxd5', 'exd5', 'cxd5', 'exd5', 'cxd5', 'exd5', 'cxd5', 'exd5', 'cxd5', 'exd5', 'cxd5', 'exd5', 'cxd5', 'exd5', 'cxd5', 'exd5', 'EOS']

Clearly, the model learns the first several moves as the games, but soon, it quickly makes illegal moves, the first being 'Be7'——the Black bishop moves to a square the black knight is already on, as shown by the earlier 'Ne7.' We can also see that it starts to hallucinate many captures of 'exd5' and 'cxd5', a sign that our model is receiving nowhere close to the amount of training data it needs to be decent.

The second dataset, and the main one that I am interested in training the model for, is an enormous dataset of ~500,000 games between "Elite" players, which is defined by players with ratings greater than 2300. Specifically, I chose the December 2021 [dataset](https://database.nikonoel.fr/)——thank you @NikoNoel for this!

In the future, once I'm able to finish training the second dataset, I want to use an even bigger dataset, with at least 1 million games. The reason for this is that Transformers are very data hungry, which is a perfect transition to the next section.

## Model Architecture

The GPT architecture may seem overwhelming at first, but we'll build from the ground up, starting with the original transformer. I replicated the transformer architecture from the AttentionIsAllYouNeed paper, with a few changes I will talk about later. Here is a quick overview from the original paper itself (Vaswani et al., 2017):

![transformer](https://github.com/user-attachments/assets/44727288-9fa3-46cb-9f09-aad00895f75a)

As can be seen, there are two parts to the Transformer Architecture——the encoder (left) and decoder (right). The original purpose of the Transformer was for machine translation, so the encoder would deal with encoding a sentence in french for example, pass this encoding to the decoder, which would then produce the translation of that sentence in english. Furthermore, the Decoder would interact with the Encoder through Cross-Attention, but in my implementation, I did not need it so it is not discussed here.

Since I am only trying to generate chess games, and emulating GPT-2, I actually only need to use the Decoder, AKA a Decoder-only Transformer (which is what Chat-GPT is). However, in the future, I can consider using an encoder to encode the current chess position, which could potentially help the decoder generate even better games. For that reason, I implemented both the Encoder and Decoder and will talk about both.

We will start with the first step, going from tokens——which we got from the tokenizer——to embeddings. We want to represent each unique token (AKA chess move) with a vector that is learned through training, and we do this through a simple PyTorch Embedding table:

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
            out = self.token_embedding(x)
            return out

Now, we have a way to represent each possible chess move with an embedding, however it's clear that playing a move like d4 on the first move of the game is very different from playing d4 on move 20, so how can our transformer model capture differences in token position? We generate a Positional Embedding for each token.

    class PositionalEmbedding(nn.Module):

        """
        Class that provides embedding that captures the positional information of a token.
        """

        def __init__(self, block_size, emb_dim):
            super(PositionalEmbedding, self).__init__()
            self.block_size = block_size
            self.emb_dim = emb_dim
        
        def forward(self, x):
            # Use torch.no_grad to make sure PositionalEmbedding not part of computational graph
            with torch.no_grad():
                # Credit to https://github.com/hyunwoongko/transformer 

                pos_embedding = torch.zeros(self.block_size, self.emb_dim)
                position = torch.arange(self.block_size).unsqueeze(dim=1)
                _2i = torch.arange(0, self.emb_dim, step=2)
                pos_embedding[:, 0::2] = torch.sin(position / (10000 ** (_2i / self.emb_dim)))
                pos_embedding[:, 1::2] = torch.cos(position / (10000 ** (_2i / self.emb_dim)))
           
            return pos_embedding

The original AttentionIsAllYouNeed paper covered two possible ways of doing the PositionalEmbedding——learned (using neural networks) and the one I used, sinusoidal positional embeddings (not learned). I have to give credit to @hyunwoongko for his implementation of the sinusoidal positional embedding, which was tricky to wrap my head around at first. Thank you to [Amirhossein Kazemnejad](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/) and [Mehreen Saeed's](https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/) blogs for helping me understand this crucial concept——feel free to check them out for more information. 

To combine both the positional and token embedding information for each token, we just add the two together:

    x = self.tok_emb(x) + self.pos_emb(x)

Now that we have the token representations, they are ready to be passed into the Encoder/Decoder. Let's start with the Encoder first, but keep in mind that the components of the Encoder and Decoder are very similar. 

### LayerNorm

LayerNorm is a method to normalize the features of each token independently. This helps stabilize training, and in the original AttentionIsAllYouNeed paper, they applied the LayerNorm after the Attention layer (we will get to what Attention is shortly). However, after the [PreNorm paper](https://arxiv.org/pdf/2002.04745), the PreNorm method has been favored, so I applied LayerNorm right after the Embeddings and before Attention. Thank you Andrej Karpathy and specifically this [video](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5878s) for being a great guide for understanding PreNorm and the overall model building process. 

Fortunately, applying LayerNorm is trivial in PyTorch: 

    self.ln_1 = nn.LayerNorm(model_dim)
    ...

    def forward(self, x):
       
        x_norm = self.ln_1(x)

After applying LayerNorm, we are now at the stage of Attention.

### Attention

At the heart of the Transformer architecture is the attention mechanism, which allows the tokens to "talk with each other." In other words, attention allows tokens from any position to incorporate their information with tokens in other positions in the same sequence. 

There are many amazing explanations of attention on the Internet or in college courses, but I'll include a brief, intuitive way of how I understand it, having learned it from so many different people. For sake of clarity, let's leave the chess world for a moment and pretend we are calculating attention on a english sentence like: 

Cherries are Hairan's favorite fruit 

It's clear to anyone reading the sentence that the words "Cherries" and "fruit" along with "Hairan's" and "favorite" are very related to each other in this phrase. Ideally, when we train our model——let's say it was a GPT model——hopefully it would learn these relations so that when it generates sentences after this one, it can make use of these connections to create more coherent and logical sentences. But, how does our GPT model learn these relations? That's where Queries, Keys, and Values——the foundations of attention——come into play.

Queries: Think of these as a token asking for what it's looking for, similar to how you might query a database in SQL. Each token will have a unique query, which will depend on its own characteristics. This is as far as I can explain it, and it is very high-level, which is mostly because interpretabililty around Transformers is an ongoing research problem——we don't know a lot about why they are so powerful. 

To get the queries for each token, we simply use a PyTorch linear layer that goes from the original embeddings we got after the positional + token embeddings.

    self.Q = nn.Linear(model_dim, model_dim)

Now that each token has queries to "ask" the other tokens, we need the other tokens——including the current token itself——to provide an "answer," so we can see if the two tokens are related to each other, or not so much. This gets us to Keys.

Keys: These are the answers to our queries, and again each token will have its own key vector. Similar to Queries, we use a simple PyTorch Linear layer:

    self.K = nn.Linear(model_dim, model_dim) 

Now, how can we have each token use its query to "ask" the rest of the token's keys to get an "answer?" A simple tensor multiplication of the queries with the *transpose* of the keys. If we had x keys, and x queries, each with dimension y, and ignoring the batch dimension, this is just a matrix multiplication of (x, y) @ (y, x) --> (x, x), where entry (i, j) in the matrix is the result of the i'th query and j'th key——in other words the result of how important the j'th token is to the i'th token. Notice that this is not a symmetrical relationship: the attention value of (i,j) is not necessarily equal to (j, i). 

    pre_softmax = (q @ k.transpose(-1, -2)) / (model_dim ** 0.5) 

Now that we have these attention values for each token, we still need to weigh the importance of the other tokens (and itself!) to produce the final representations for each token. Notice the name of the above variable is pre-softmax, since we will soon be taking the softmax of it. To get the final token representations through this attention layer, we use the Values and softmax. 

Values: Pretty self-explanatory——they represent the values of the tokens themselves. These values are *not* the final representations——they will need to be weighted based on a softmax distribution of how important the other tokens are to the current token. This softmax distribution is produced with the matrix we generated above after doing q @ k.transpose:

    post_softmax = self.softmax(pre_softmax)
        
    out = post_softmax @ v
    
    return out 

It's important to note that the attention calculated in the encoder will differ from attention calculated in the decoder because we will not want early tokens to attend to later tokens since that would be "cheating", but we'll get more into the details of that difference later on. For now on, let's move on to the next stage of the model, the residual connection.

### Residual Connection 

The GPT model we will be training is going to have many, many layers. And, while this will help the model learn more complex representations of data, the potential drawback is that during training, gradients will get smaller and smaller as they move from layer to layer, eventually vanishing completely and hindering training since we now have dead neurons. Fortunately, we can make use of Residual Connections, an idea published by He et al. in 2015. These residuals act as a gradient highway, allowing the gradient to flow from the loss directly back to the input embeddings. This way, even if early on in training, there becomes dead neurons within the attention calculations, we can still continue training because of the residual connections. Implementing this in code is trivial: 

    x = x + attention # first residual (the norm is inside the residual)

This is one example of a residual, where the attention variable represents the value of the tokens after LayerNorm and attention, and x is before the attention. 

### Position-wise Feedforward Network

After the attention calculations, we have a FeedForward layer that acts on each position of the sequence independently. Essentially, this 2-layer linear neural network further helps each token incorporate the information it learned from the attention layer. We do the same process of layer norm to normalize each position's weights to aid training as well as the residual connection to prevent vanishing gradients. 

    self.ffn = FeedForwardNetwork(model_dim, model_dim)

Now after this Feed Forward Network, this completes one Encoder Block. Since Transformers and GPT models are made up of many such Encoder/Decoder Blocks, all we have to do is pass the output of the first Encoder block into the next block, eventually until we reach the final block.

This is a perfect time to transition to talking about the Decoder, since it has a unique final layer, where it must map the output from the final Decoder Block into probabilities for the next token. This way, using our decoder (which is what ChatGPT relies on), we can probabilistically generate the next token. As an example, for:

Cherries are Hairan's favorite fruit

the next token generate could be "because", and after that maybe we would get a sentence of:

Cherries are Hairan's favorite fruit because he can eat so many at a time.

All of this is to say, we need a way for our model to generate a prediction for the next token, which means we must map our model's current output, which is in the model_dim, to the dimension of vocab_size, to generate a probability distribution we can sample from.

### Projection Head

Our projection head maps from the model dimension to the size of our vocabulary, which is an easy PyTorch Linear layer combined with a softmax to generate our probability distribution:

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

With this final output, we can then generate the next token, and we'll go more into depth about this when we talk about Inference. 

For now, those are the main components for the Encoder and Decoder blocks. We will now talk about the significant difference between the attention calculation between the Encoder and Decoder: masking.

### Attention Masking

To give intuition on why we need different ways of calculating attention for Encoder versus Decoder, let's go back to the 



Now, we will get into slightly more advanced aspects of the code, some of which aren't technically necessary but have shown to lead to better results and are implemented in the AttentionIsAllYouNeed paper. We will the

### 

## Training

Talk about produce_pairs function to make it efficient training.

**Still training on the largest dataset with the help of GPUs, so this section will be updated once done with training.

## FastAPI

## Docker 




