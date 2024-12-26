"""
The inference module allows a user to pass in a path to config file with model hyperparameters, path to trained model,  
number of times they want the model to generate moves, and optionally the moves for each generation.

Ex: <path_to_config> <path_to_model> 
"""
from model import Decoder
import yaml
import pickle
import torch
import sys
import argparse

def create_parser():
    parser = argparse.ArgumentParser(
        description='''Chess game generation',
    Examples:
    # Generate 5 games from scratch:
    python inference.py config.yaml model.pt 5
    
    # Generate 3 games starting with specific moves:
    python inference.py config.yaml model.pt 3 -m "e4 e5;d4 d5"
    
    # Generate 2 games with moves from a file:
    python inference.py config.yaml model.pt 2 -f moves.txt
    
    # Generate 4 games with specific random seed:
    python inference.py config.yaml model.pt 4 -s 42''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('config_path', help='Path to config file containing model hyperparameters')
    parser.add_argument('model_path', help='Path to trained model')
    parser.add_argument('num_generations', type=int, help='Number of games to generate')

    parser.add_argument('--moves-file', '-f', 
                        help='File containing starting moves (one sequence per line)')
    parser.add_argument('--moves', '-m',
                        help='Starting moves in compact notation (sequences separated by semicolons)')
    parser.add_argument('--random-seed', '-s', type=int,
                        default=None,
                        help='Random seed for reproducible generation')

    return parser
    

def generate(model, token_dict, max_game_length, decode_dict, starting_moves=None):

    """
    Takes trained model, token dictionary, max_game_length
    """

    EOS_token = len(token_dict) - 1 # EOS token is last token

    model.eval() # Set to evaluation mode for inference

    # Initialize game to be generated
    game_tokens = [token_dict['BOS']]
    # Only add onto the game if starting moves not None
    if starting_moves != None:
        for move in starting_moves:
            game_tokens.append(token_dict[move])

    # Now, we just want to feed it to the model continuously, until the token we are predicting gives us EOS or reach max_game_length
    while len(game_tokens) < max_game_length:
        input_tensor = torch.tensor(game_tokens).unsqueeze(0)  # Shape: (1, sequence_length)
        
        output = model(input_tensor).squeeze(0)  # Shape: (sequence_length, vocab_size)
        
        probs = output[-1]  # Get probabilities for the current last token
        
        chosen_token = torch.multinomial(probs, num_samples=1).item()
        
        game_tokens.append(chosen_token)
        if chosen_token == EOS_token:
            break
    
    notation = [decode_dict[token] for token in game_tokens
                if token not in [token_dict['PAD'], token_dict['BOS']]]
    return notation

def main():

    parser = create_parser()    # Parsing command line arguments
    args = parser.parse_args() 
    
    model_path = args.model_path
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Dataset information
    game_list_path = config['game_list']
    token_dict_path = config['token_dict']
    decode_dict_path = config['decode_dict']
    game_counter_path = config['game_counter']

    # Model parameters
    block_size = config['block_size']
    num_dec_blocks = config['num_dec_blocks']
    emb_dim = config['emb_dim']
    model_dim = config['model_dim']
    num_heads = config['num_heads']

    # Inference parameters
    max_game_length = config['max_game_length']

    game_list = []
    token_dict = {} # This will act as the tokenizer/encoder
    decode_dict = {} # This will act as the decoder from tokens to string/SAN 
    game_counter = 0

    with open(token_dict_path, 'rb') as file:
        token_dict = pickle.load(file)

    with open(decode_dict_path, 'rb') as file:
        decode_dict = pickle.load(file)

    # Set first token to 0/'BOS'
    token_dict['BOS'] = 0 
    decode_dict[0] = 'BOS'

    pad_token = len(token_dict)
    token_dict['PAD'] = pad_token # Adding in a padding token
    decode_dict[pad_token] = 'PAD'

    eos_token = len(token_dict)
    token_dict['EOS'] = eos_token # Adding in an EOS token
    decode_dict[eos_token] = 'EOS'

    vocab_size = len(token_dict)

    print(f"Number of tokens, including special tokens: {vocab_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Decoder(vocab_size=vocab_size, 
                    block_size=block_size, 
                    num_dec_blocks=num_dec_blocks, 
                    emb_dim=emb_dim, 
                    model_dim=model_dim, 
                    num_heads=num_heads).to(device) # Moving model to the device

    state_dict = torch.load(args.model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total number of parameters: {total_params}")

    starting_moves_list = []    # This is a list of the starting moves for the chess games
    if args.moves_file:
        with open(args.moves_file, 'r') as f:
            for line in f:
                moves = line.strip().split()
                if moves:
                    starting_moves_list.append(moves)
            starting_moves = f.readline.strip().split()

    elif args.moves:
        sequences = args.moves.split(';')
        for seq in sequences:
            moves = seq.strip().split()
            if moves:
                starting_moves_list.append(moves)
    
    # If no starting moves, generate from scratch
    if not starting_moves_list:
        starting_moves_list = [None] * args.num_generations
    
    for i in range(args.num_generations):
        print(f"\nGenerating game {i+1}/{args.num_generations}")

        starting_moves = starting_moves_list[i % len(starting_moves_list)]
        game = generate(
            model=model,
            token_dict=token_dict,
            max_game_length=max_game_length,
            decode_dict=decode_dict,
            starting_moves=starting_moves
        )
        print(f"Game {i+1} with starting moves {starting_moves}: {' '.join(game)}")

if __name__ == '__main__':
    main()


# Note to self: Will have to fix the code for tokenizer.py and inference.py——bos=0, pad=1, eos=2

