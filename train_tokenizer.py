#! pip install tokenizers

from pathlib import Path

from tokenizers import ByteLevelBPETokenizer, BertWordPieceTokenizer
from transformers import BertTokenizerFast
import sys

paths = ["data/malware_sentences_raw.txt"]

model = "bert"

if model == "bert":
    tokenizer = BertWordPieceTokenizer()

    # sys.exit()
    # Customize training
    tokenizer.train(files=paths, vocab_size=10000, min_frequency=2,
    special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'], wordpieces_prefix="##", show_progress=True
    )

else:
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=paths, vocab_size=10000, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

# Save files to disk
tokenizer.save_model(".", model)
