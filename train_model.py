import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataset
 
 
from argparse import ArgumentParser
from torchscale.architecture.config import DecoderConfig, RetNetConfig, EncoderDecoderConfig
from torchscale.architecture.decoder import Decoder
from torchscale.architecture.retnet import RetNetDecoder
from torchscale.architecture.encoder_decoder import EncoderDecoder
 
 
from torchinfo import summary as model_summary
 
 
from datasets import load_translation_text
 
 
from tqdm import tqdm
 
 
from tabulate import tabulate
import os


class EncoderDecoderModel(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            value_embed_dim: int,
            attention_heads: int,
            ffn_dim: int,
            layers: int,
            dropout: float,
            activation_dropout: float,
            vocab_size: int,
            checkpoint_activations: bool,
            fsdp: bool,
            max_seq_len: int):
        super().__init__()
 
        tokenizer_path = "models/tokenizer" + '_' + str(vocab_size)
 
        self.params = {
                "embed_dim": embed_dim,
                "value_embed_dim": value_embed_dim,
                "attention_heads": attention_heads,
                "ffn_dim": ffn_dim,
                "layers": layers,
                "dropout": dropout,
                "activation_dropout": activation_dropout,
                "vocab_size": vocab_size,
                "checkpoint_activations": checkpoint_activations,
                "fsdp": fsdp,
                "max_seq_len": max_seq_len,
                "tokenizer_path": tokenizer_path
                }
 
 
        config = EncoderDecoderConfig(
                encoder_embed_dim=embed_dim,
                decoder_embed_dim=embed_dim,

                decoder_value_embed_dim=value_embed_dim,

                encoder_attention_heads=attention_heads,
                decoder_attention_heads=attention_heads,

                encoder_ffn_embed_dim=ffn_dim,
                decoder_ffn_embed_dim=ffn_dim,
                
                encoder_layers=layers,
                decoder_layers=layers,
                dropout=dropout,
                activation_dropout=activation_dropout,
                vocab_size=vocab_size,
                checkpoint_activations=checkpoint_activations,
                fsdp=fsdp)
 
 
        # Save max_seq_len for padding later
        self.max_seq_len = max_seq_len
 
 
        # Save vocab_size for final dimensions later
        self.vocab_size = vocab_size
 
 
        # Create embeddings with index 0 representing padding
        self.text_embeddings = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embed_dim,
                padding_idx=0)
 
 
        self.model = EncoderDecoder(config, encoder_embed_tokens=self.text_embeddings, decoder_embed_tokens=self.text_embeddings)
 
 
    def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, other_stuff = self.model(x, targets)
        return logits
   
    def generate_text(self, src, device='cuda'):
        # Start of translated text
        tgt_string = "[BOS]"

        # Evaluation mode
        self.model.eval()
        self.model.to(device)

        # Convert source string to numbers
        src = self.tokenizer.encode(src)
        src = torch.tensor(src).unsqueeze(0).to(device)
 
        # Convert start string to numbers
        tgt = self.tokenizer.encode(tgt_string)
        tgt = torch.tensor(tgt).unsqueeze(0).to(device)
 
        # No gradients needed
        with torch.no_grad():
            for i in range(self.max_seq_len):
                predictions = self.forward(src, tgt)
                # Apply softmax to get probabilities
                predictions = F.softmax(predictions, dim=-1)

                # Get the last predicted word
                predicted_id = predictions.argmax(dim=-1)[..., -1]

                # Stop if predicted word is the end token
                if predicted_id == self.tokenizer.encode("[EOS]"):
                    break
 
                # Add predicted word to the target (to be used as next input sequence)
                tgt = torch.cat([tgt, predicted_id.unsqueeze(-1)], dim=-1)
 
 
        return self.tokenizer.decode(tgt.squeeze().tolist())
 
if __name__ == "__main__":
    # Initialize, setup, and parse the argument parser
    parser = ArgumentParser(
            prog="Model Trainer",
            description="Used to train comparable RetNet, Transformer models.")
 
 
    parser.add_argument("-a", "--activation-dropout", type=float, default=0.0,
            help="Probability of element to be zeroed in dropout layer " + \
                    "after activation between FFN layers.")
    parser.add_argument("-c", "--checkpoint-activations", type=bool,
            default=False, help="Use checkpointing.")
    parser.add_argument("-d", "--dropout", type=float, default=0.0,
            help="Probability of element to be zeroed in dropout layer.")
    parser.add_argument("-e", "--embed-dim", type=int, default=768,
            help="Embedding dimension size of each token.")
    parser.add_argument("-f", "--ffn-dim", type=int, default=1280,
            help="FFN hidden layer size.")
    parser.add_argument("--fsdp", type=bool, default=False,
            help="Module parameters sharded across data parallel workers.")
    parser.add_argument("-l", "--layers", type=int, default=12,
            help="Number of stacked layers in model.")
    parser.add_argument("--lr", type=float, required=True,
            help="Learning rate of model to train.")
    parser.add_argument("-m", "--model", required=True,
            choices=["enc_dec"],
            help="Name of model architecture to train.")
    parser.add_argument("-n", "--heads", type=int, default=3,
            help="Number of heads. Head architecture changes based on model.")
    parser.add_argument("-s", "--seq-len", type=int, default=512,
            help="Sequence length (context window size).")
    parser.add_argument("--value-embed-dim", type=int, default=1280,
            help="Value embed dimension size.")
    parser.add_argument("--vocab-size", type=int, required=True,
            help="Maximum number of unique tokens in vocabulary.")
    parser.add_argument("--batch-size", type=int, default=32,
            help="Batch size.")
    parser.add_argument("--device", type=str, default='cuda',
            help="Device to use (GPU).")
    parser.add_argument("--epochs", type=int, default=10,
            help="Number of epochs to train for.")
    parser.add_argument("--name", type=str, default="test",
            help="Name of the test run.")
 
 
    args = parser.parse_args()
   
    # Test that the head dimension will be an even, whole number
    assert args.embed_dim % (args.heads * 2) == 0, \
            "Head Dimension must be even to perform Rotary Position " + \
            f"Embedding ({args.embed_dim} / {args.heads} = " + \
            f"{args.embed_dim / args.heads} -- not an even, whole number)! " + \
            "Try changing the Embedding Dimension or number of heads."
 
 
    # Test that the value embedding dimension is divisible by number of heads
    assert args.value_embed_dim % args.heads == 0, \
            "Value Embed Dimension not divisible by number of heads " + \
            f"({args.value_embed_dim} % {args.heads} != 0)!"
 
 
    # Create requested model
    if args.model == "enc_dec":
        model = EncoderDecoderModel(
                embed_dim=args.embed_dim,
                value_embed_dim=args.value_embed_dim,
                attention_heads=args.heads,
                ffn_dim=args.ffn_dim,
                layers=args.layers,
                dropout=args.dropout,
                activation_dropout=args.activation_dropout,
                vocab_size=args.vocab_size,
                checkpoint_activations=args.checkpoint_activations,
                fsdp=args.fsdp,
                max_seq_len=args.seq_len)
 
    # Print all arguments for recordkeeping
    print('Arguments:')
    arg_table = []
    row = []
    for i, arg in enumerate(vars(args)):
        row.append(f'{arg}: {getattr(args, arg)}')
        if (i + 1) % 4 == 0:
            arg_table.append(row)
            row = []
    if row:
        arg_table.append(row)
 
 
    print(tabulate(arg_table, tablefmt="grid"))
 
 
    # Print model info
    # print('\nModel Summary:')
    # model_summary(model, (args.batch_size, args.seq_len), (args.batch_size, args.seq_len))
 
 
    # Print estimated loss if it hasn't learned anything
    print('\nEstimated Loss if guessing:')
    print(f'-log(1 / {args.vocab_size}) = {-torch.log(torch.tensor(1 / args.vocab_size))}')
 
 
    # Load the dataset
    train_loader, valid_loader, test_loader, tokenizer = load_translation_text(max_seq_len=args.seq_len, batch_size=args.batch_size, vocab_size=args.vocab_size)
    model.tokenizer = tokenizer
 
 
    # Define loss function
    loss_fn = nn.CrossEntropyLoss()
 
 
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
 
 
    # Define the device to use
    device = torch.device(args.device)
 
 
    # Put model on device
    model = model.to(device)
 
 
    # Train the model
    print('\nTraining model...')
    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}')
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, mininterval=60)): # Prints progress bar every mininterval seconds
            
            # Train an encoder-decoder model
 
            # Put inputs and targets on device
            inputs = inputs.to(device)
            targets = targets.to(device)
       
            # Zero out gradients
            optimizer.zero_grad()
       
            # Get model predictions
            predictions = model(inputs, targets[:, :-1])
           
            # Reshape the model outputs to match the expected shape for CrossEntropyLoss
            predictions = predictions.view(-1, predictions.size(-1))
            targets = targets[:, 1:].contiguous().view(-1)
            loss = loss_fn(predictions, targets)
       
            # Backpropagate loss
            loss.backward()
       
            # Update parameters
            optimizer.step()
 
            # Run validation 3 times per epoch
            if batch_idx % (len(train_loader) // 3) == 0:
                # Print train loss
                print(f"Train Loss: {loss.item()}")
                model.eval()
                with torch.no_grad():
                    total_loss = 0
                    total_samples = 0
                    for val_inputs, val_targets in valid_loader:
                        # Put validation inputs and targets on device
                        val_inputs = val_inputs.to(device)
                        val_targets = val_targets.to(device)
                       
                        # Get validation predictions
                        val_predictions = model(val_inputs, val_targets[:, :-1])
                       
                        # Calculate validation loss
                        val_predictions = val_predictions.view(-1, val_predictions.size(-1))
                        val_targets = val_targets[:, 1:].contiguous().view(-1)
                        val_loss = loss_fn(predictions, targets)
                        total_loss += val_loss.item() * val_inputs.size(0)
                        total_samples += val_inputs.size(0)
                   
                    # Calculate average validation loss
                    avg_val_loss = total_loss / total_samples
                    print(f"Validation Loss: {avg_val_loss}")
               
                model.train()
 
 
    # Test the model
    print('\nTesting model...')
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, mininterval=60): # Prints progress bar every mininterval seconds
            # Put inputs and targets on device
            inputs = inputs.to(device)
            targets = targets.to(device)
           
            # Get model predictions
            predictions = model(inputs, targets[:, :-1])
           
            # Calculate loss
            predictions = predictions.view(-1, predictions.size(-1))
            targets = targets[:, 1:].contiguous().view(-1)
            loss = loss_fn(predictions, targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
   
    # Calculate average loss
    avg_loss = total_loss / total_samples
    print(f"Test Loss: {avg_loss}")
 
 
    # Generate text from the model
    print('\nGenerating text...')
    print(model.generate_text(src="Hi, my name is Anthony!", device=device))
    print(model.generate_text(src="What is your name?", device=device))
    print(model.generate_text(src="What is your favorite color?", device=device))
    print(model.generate_text(src="The brothers' business was a success and they became more active in civic affairs, both in Philadelphia and the wider field of the colony of Pennsylvania.", device=device))

    # Create directory for model if it doesn't exist
    if not os.path.exists(os.path.join("models", args.name)):
        os.makedirs(os.path.join("models", args.name))

    # Save the model
    print('\nSaving model...')
    torch.save(model.state_dict(), os.path.join("models", args.name, "model.pt"))

    # Pickle the model's parameters
    print('\nSaving model parameters...')
    torch.save(model.params, os.path.join("models", args.name, "params.pkl"))
