import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import sentencepiece as spm
import random
import os

def create_tokenizer(src_lines, tgt_lines, vocab_size):
    # Parameters
    sample_fraction = 0.1  # for example, 10% of the data

    # Function to write a subset of the dataset to a file
    def write_subset_to_file(train_src, train_tgt, filename, fraction):
        combined_data = [(src.strip(), tgt.strip()) for src, tgt in zip(train_src, train_tgt)]
        sampled_data = random.sample(combined_data, int(len(combined_data) * fraction))

        with open(filename, "w") as f:
            for src, tgt in sampled_data:
                f.write(src + "\n" + tgt + "\n")

    # Create a folder to store the tokenizer and the datasets
    tokenizer_path = "models/tokenizer" + '_' + str(vocab_size)
    if not os.path.exists(tokenizer_path):
        os.makedirs(tokenizer_path)

    # Write out a subset of the train set to datasets/train.txt
    write_subset_to_file(src_lines, tgt_lines, "datasets/train.txt", sample_fraction)

    # Create a custom sentencepiece tokenizer from the train subset file
    num_defined_tokens = 2 # [BOS] and [EOS]
    spm.SentencePieceTrainer.train(
        input="datasets/train.txt",
        model_prefix=os.path.join(tokenizer_path, "tokenizer"),
        vocab_size=vocab_size-num_defined_tokens,
        user_defined_symbols=['[BOS]', '[EOS]'],
        pad_id=3
    )

def make_dataloader(src_lines, tgt_lines, tokenizer, max_seq_len, batch_size, shuffle=False):
    # Function to process each article
    def data_process(raw_text_iter):
        processed_data = []
        for text in raw_text_iter:
            # Tokenize and numericalize
            numericalized_text = tokenizer.encode(text)
            # Pad and possibly truncate the sequence
            padded = F.pad(torch.tensor(numericalized_text, dtype=torch.long),
                          (0, max_seq_len - len(numericalized_text)),
                          value=tokenizer.pad_id())
            if len(padded) > max_seq_len:
                padded = padded[:max_seq_len]
            processed_data.append(padded)
        return processed_data

    # Process the datasets
    src_data = data_process(src_lines)
    tgt_data = data_process(tgt_lines)

    # Custom Dataset class
    class TranslationDataset(Dataset):
        def __init__(self, src_data, tgt_data):
            self.src_data = src_data
            self.tgt_data = tgt_data

        def __len__(self):
            return len(self.src_data)

        def __getitem__(self, idx):
            src_item = self.src_data[idx]
            tgt_item = self.tgt_data[idx]
            # Ensure the sequence is of MAX_SEQ_LEN
            src_item_padded = F.pad(src_item, (0, max_seq_len - len(src_item)), value=tokenizer.pad_id())
            tgt_item_padded = F.pad(tgt_item, (0, max_seq_len - len(tgt_item) + 1), value=tokenizer.pad_id()) # Add 1 so we can index targets two different ways
            # Input is the entire sequence
            input = src_item_padded
            # Target is the same sequence shifted by one position and padded
            target = F.pad(tgt_item_padded[1:], (0, 1), value=tokenizer.pad_id())
            return input, target

    # Create datasets
    dataset = TranslationDataset(src_data, tgt_data)

    # DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader


def load_translation_text(max_seq_len, batch_size, vocab_size):
    """ Loads the malagasy translation dataset and returns the train, validation and test data loaders
    Args:
        max_seq_len (int): Maximum sequence length
        batch_size (int): Batch size
    Returns:
        train_loader (DataLoader): Training data loader
        valid_loader (DataLoader): Validation data loader
        test_loader (DataLoader): Test data loader
    """
    # Load the dataset
    src_files = [
        "datasets/Malagasy/cleaned/src.txt",
        "datasets/mg/cleaned/src.txt",
    ]
    tgt_files = [
        "datasets/Malagasy/cleaned/tgt.txt",
        "datasets/mg/cleaned/tgt.txt",
    ]

    # Set the torch random seed
    torch.manual_seed(0)

    # Create a combined dataset
    src_lines = []
    tgt_lines = []

    # Do pwd to get the current directory
    current_directory = os.getcwd()
    print(current_directory)

    # Go up one directory
    os.chdir("..")

    for src_file, tgt_file in zip(src_files, tgt_files):
        with open(src_file, "r") as f:
            src_lines.extend(f.readlines())
        with open(tgt_file, "r") as f:
            tgt_lines.extend(f.readlines())

    # Shuffle the dataset
    indices = torch.randperm(len(src_lines))
    src_lines = [src_lines[i] for i in indices]
    tgt_lines = [tgt_lines[i] for i in indices]

    # Make test, validation and training splits
    test_split = 0.01
    valid_split = 0.01
    test_size = int(len(src_lines) * test_split)
    valid_size = int(len(src_lines) * valid_split)
    train_size = len(src_lines) - test_size - valid_size

    # Split the lists into test, validation and training sets
    test_src = src_lines[:test_size]
    test_tgt = tgt_lines[:test_size]
    valid_src = src_lines[test_size:test_size + valid_size]
    valid_tgt = tgt_lines[test_size:test_size + valid_size]
    train_src = src_lines[test_size + valid_size:]
    train_tgt = tgt_lines[test_size + valid_size:]

    # Put '[BOS]' and '[EOS]' tokens in the target sequences
    train_tgt = ['[BOS] ' + tgt + ' [EOS]' for tgt in train_tgt]
    valid_tgt = ['[BOS] ' + tgt + ' [EOS]' for tgt in valid_tgt]
    test_tgt = ['[BOS] ' + tgt + ' [EOS]' for tgt in test_tgt]

    # If there isn't a tokenizer.model and tokenizer.vocab in models/tokenizer, create them
    tokenizer_path = "models/tokenizer" + '_' + str(vocab_size)
    tokenizer_model_path = os.path.join(tokenizer_path, "tokenizer.model")
    tokenizer_vocab_path = os.path.join(tokenizer_path, "tokenizer.vocab")
    if not os.path.exists(tokenizer_model_path) or not os.path.exists(tokenizer_vocab_path):
        create_tokenizer(train_src, train_tgt, vocab_size)

    # Load the tokenizer
    tokenizer = spm.SentencePieceProcessor(model_file=os.path.join(tokenizer_path, "tokenizer.model"))

    # Make dataloaders
    #!: Fix this
    train_loader = make_dataloader(train_src, train_tgt, tokenizer, max_seq_len, batch_size, shuffle=False)
    valid_loader = make_dataloader(valid_src, valid_tgt, tokenizer, max_seq_len, batch_size)
    test_loader = make_dataloader(test_src, test_tgt, tokenizer, max_seq_len, batch_size)

    return train_loader, valid_loader, test_loader, tokenizer

# train_loader, valid_loader, test_loader, tokenizer = load_translation_text(128, 32, 10000)

# for batch in train_loader:
#     print(batch)
