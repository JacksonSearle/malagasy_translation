import os
import torch
import sentencepiece as spm
from train_model import EncoderDecoderModel


# Go up one directory
os.chdir("..")

# Generate text from a saved model
models_folder = "models"
model_folder = "test"

# Load model.pt
model = torch.load(os.path.join(models_folder, model_folder, "model.pt"))

# Load params.pkl
params = torch.load(os.path.join(models_folder, model_folder, "params.pkl"))

# Instantiate the model
enc_dec = EncoderDecoderModel(params["embed_dim"], params["value_embed_dim"], params["attention_heads"], params["ffn_dim"],
                              params["layers"], params["dropout"], params["activation_dropout"], params["vocab_size"], 
                              params["checkpoint_activations"], params["fsdp"], params["max_seq_len"])

# Load the model's state_dict
enc_dec.load_state_dict(model)

# Load the vocabulary
tokenizer = spm.SentencePieceProcessor(model_file=os.path.join(params["tokenizer_path"], "tokenizer.model"))
enc_dec.tokenizer = tokenizer

# Generate text
print(enc_dec.generate_text(src='He wants us to return to Him.', device='cuda'))
print(enc_dec.generate_text(src='He knows us personally.', device='cuda'))
print(enc_dec.generate_text(src='He knows you.', device='cuda'))


# Print the number of model parameters
def count_parameters(model):
    """
    Count the number of parameters in a PyTorch model.

    Args:
    model (nn.Module): A PyTorch model.

    Returns:
    int: Total number of parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
