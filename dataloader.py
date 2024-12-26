from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import chess.pgn
import matplotlib.pyplot as plt
import numpy as np
from util import produce_pairs
import torch
import pickle

# Design choice: have the dataloader call the utility function which then handles the input target pairs here

# What will be fed in is a list of moves, for which all that has to be done is convert them to the token through a simple for loop
class ChessDataset(Dataset):
    def __init__(self, game_list, token_dict, max_game_length):
        # print("initialized")
        self.game_list = game_list
        self.token_dict = token_dict # This will just be a dictionary mapping from chess move to integer(Long dtype)
        self.max_game_length = max_game_length + 1 # Adjustment for special tokens

    def __len__(self):
        # print("inside len")
        # print(f"len: {len(self.game_list)}")
        return len(self.game_list)

    def __getitem__(self, idx):
        # print("inside getitem")
        # Remember, this is applied to one element of the batch, it doesn't see the other batch elements here
        prepend = 'BOS'
        
        game = self.game_list[idx].copy() # Have to make sure don't mutate it with each access, so just make a copy
        # print(f"game before: {game}")
        game = [prepend] + game

        # Got my logic all wrong! Need to have the EOS token first, then have the pad tokens come AFTER to fill out the rest

        # print(f"game: {game}")
        # Now, need to append the PAD token to the end of the tokenized game: if len() < max_length + 1, then add until it is max_length + 1
        if len(game) >= self.max_game_length:
            game = game[:self.max_game_length - 1]
            game.append('EOS')
        else:
            game.append('EOS')
            while len(game) < self.max_game_length:
                game.append('PAD')

        # print(f"game: {game}")
        tokenized_game = self.tokenize(game, self.token_dict)

        # print(f"Tokenized game: {tokenized_game}")
        # print(f"shape of tokenized game: {tokenized_game.shape}")

        PAD_token = len(self.token_dict) - 2 # PAD token is second last token
        EOS_token = len(self.token_dict) - 1 # EOS token is last token
        loss_mask = torch.tensor([0 if token == PAD_token else 1 for token in tokenized_game])

        # print(f"PAD_token: {PAD_token}")
        # print(f"EOS_token: {EOS_token}")
        # print(f"token_dict: {token_dict}")
        # print(f"tokenized_game: {tokenized_game}")
        # print(f"original loss_mask: {loss_mask}")

        loss_mask = loss_mask[:-1] # This is to correct for the fact that we grabbed max_length + 1 tokens, since we are using the produce_pairs function
        # print(f"shifted loss_mask: {loss_mask}")
        # print(f"length of game: {len(game)}")
        return tokenized_game, loss_mask # loss mask used for calculating loss to ignore padded tokens; for encoder could be used to ignore padding in attention calculation I believe
    
    def tokenize(self, game, token_dict):
        # print("inside tokenize")
        # game is a list of chess moves
        for i, move in enumerate(game):
            if move not in token_dict:
                print(f"Move '{move}' not found in token_dict!")
            game[i] = token_dict[move]
        return torch.tensor(game)
                        
    
if __name__ == '__main__':

    game_list = []
    token_dict = {} # This will act as the tokenizer/encoder
    reverse_dict = {} # This will act as the decoder from tokens to string/SAN 
    game_counter = 0

    with open('LichessElite/game_list.pkl', 'rb') as file:
        game_list = pickle.load(file)

    with open('LichessElite/token_dict.pkl', 'rb') as file:
        token_dict = pickle.load(file)

    with open('LichessElite/reverse_dict.pkl', 'rb') as file:
        reverse_dict = pickle.load(file)

    with open('LichessElite/counter.pkl', 'rb') as file:
        game_counter = pickle.load(file)

    # After edits, this will all be taken care of in the token_dict

    # Set first token to 0/'BOS'
    # token_dict['BOS'] = 0 
    # reverse_dict[0] = 'BOS'

    # pad_token = len(token_dict)
    # token_dict['PAD'] = pad_token # Adding in a padding token
    # reverse_dict[pad_token] = 'PAD'

    # eos_token = len(token_dict)
    # token_dict['EOS'] = eos_token # Adding in an EOS token
    # reverse_dict[eos_token] = 'EOS'

    vocab_size = len(token_dict)
    print(f"number of games: {game_counter}")
    print(f"tokenizer dict: {token_dict}")
    print(f"decoder dict: {reverse_dict}")
    print(f"vocab_size: {vocab_size}")

    ds = ChessDataset(game_list, token_dict, max_game_length=120)

    print(f"length of ds inside main: {len(ds)}")
    print(f"first ds element inside main{ds[0]}")
    print(ds[0])
    





