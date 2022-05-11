from __future__ import annotations
import tempfile
import os
import json
import logging
from typing import Optional
from transformers import Wav2Vec2Processor, AutoConfig, AutoTokenizer, AutoFeatureExtractor

class TokenSet():
    """
    TokenSet

    Parameters
    ----------
    tokens : list[str]
        List of tokens.
    
    blank_token : Optional[str] = "<pad>"
        Blank token
    
    silence_token : Optional[str] = "|"
        Silence token
    
    unk_token : Optional[str] = "<unk>"
        Unk token
    
    bos_token : Optional[str] = "<s>"
        BOS token

    eos_token : Optional[str] = "</s>"
        EOS token

    letter_case: str
        Case mode to be applied to the transcription, can be 'lowercase', 'uppercase' 
        or None (None == keep the original letter case). Default is lowercase.

    """
    
    def __init__(self, tokens: list[str], blank_token: Optional[str] = "<pad>", silence_token: Optional[str] = "|", unk_token: Optional[str] = "<unk>", 
                 bos_token: Optional[str] = "<s>", eos_token: Optional[str] = "</s>", letter_case: str = "lowercase"):

        self.tokens = tokens
        self.blank_token = blank_token
        self.silence_token = silence_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.letter_case = letter_case

        if self.letter_case == "lowercase":
            self.tokens = [token.lower() if token not in self.special_tokens else token for token in self.tokens]
        elif self.letter_case == "uppercase":
            self.tokens = [token.upper() if token not in self.special_tokens else token for token in self.tokens]
        
        if blank_token not in tokens:
            logging.warning(f"blank_token {blank_token} not in provided tokens. It will be added to the list of tokens")
            self.tokens.append(blank_token)

        if silence_token not in tokens:
            logging.warning(f"silence_token {silence_token} not in provided tokens. It will be added to the list of tokens")
            self.tokens.append(silence_token)
        
        if unk_token not in tokens:
            logging.warning(f"unk_token {unk_token} not in provided tokens. It will be added to the list of tokens")
            self.tokens.append(unk_token)

        if bos_token not in tokens:
            logging.warning(f"bos_token {bos_token} not in provided tokens. It will be added to the list of tokens")
            self.tokens.append(bos_token)
        
        if eos_token not in tokens:
            logging.warning(f"eos_token {eos_token} not in provided tokens. It will be added to the list of tokens")
            self.tokens.append(eos_token)

        self.id_by_token = {token: i for i, token in enumerate(self.tokens)}
        self.token_by_id = {i: token for i, token in enumerate(self.tokens)}

    @property
    def blank_token_id(self):
        return self.id_by_token[self.blank_token]

    @property
    def silence_token_id(self):
        return self.id_by_token[self.silence_token]
    
    @property
    def unk_token_id(self):
        return self.id_by_token[self.unk_token]

    @property
    def bos_token_id(self):
        return self.id_by_token[self.bos_token]

    @property
    def eos_token_id(self):
        return self.id_by_token[self.eos_token]

    @property
    def non_special_tokens(self):
        return [token for token in self.tokens if token not in self.special_tokens]

    @property
    def special_tokens(self):
        return [self.blank_token, self.silence_token, self.unk_token, self.bos_token, self.eos_token]

    @property
    def size(self):
        return len(self.tokens)

    def to_processor(self, model_name_or_path: str="facebook/wav2vec2-large-xlsr-53"):

        tokens_dict = {v: i for i, v in enumerate(self.tokens)}

        with tempfile.TemporaryDirectory() as tmpdirname:

            vocab_path = os.path.join(tmpdirname, "vocab.json")

            with open(vocab_path, "w") as vocab_file:
                json.dump(tokens_dict, vocab_file)

            config = AutoConfig.from_pretrained(model_name_or_path)
            config_for_tokenizer = config if config.tokenizer_class is not None else None
            tokenizer_type = config.model_type if config.tokenizer_class is None else None

            tokenizer = AutoTokenizer.from_pretrained(
                tmpdirname,
                config=config_for_tokenizer,
                tokenizer_type=tokenizer_type,
                bos_token=self.bos_token,
                eos_token=self.eos_token,
                unk_token=self.unk_token,
                pad_token=self.blank_token,
                word_delimiter_token=self.silence_token,
                do_lower_case=False,
                # do_lower_case=self.letter_case == "lowercase",
                #TODO: fix transformers/models/wav2vec2/tokenization_wav2vec2.py:199
            )

            feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)

            return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    @classmethod
    def from_processor(cls, processor: Wav2Vec2Processor, letter_case: str = "lowercase"):

        blank_token = processor.tokenizer.pad_token
        silence_token = processor.tokenizer.word_delimiter_token
        unk_token = processor.tokenizer.unk_token
        bos_token = processor.tokenizer.bos_token
        eos_token = processor.tokenizer.eos_token
        tokens = [x for x in processor.tokenizer.convert_ids_to_tokens(range(0, processor.tokenizer.vocab_size))]

        return cls(tokens, blank_token, silence_token, unk_token, bos_token, eos_token, letter_case)

    def save(self, path: str):

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):

        with open(path, encoding="utf-8") as f:
            o = json.load(f)
            return cls(o["tokens"], o["blank_token"], o["silence_token"], o["unk_token"], o["bos_token"], o["eos_token"], o["letter_case"])