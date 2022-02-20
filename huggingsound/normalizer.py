from __future__ import annotations
import re
from huggingsound.token_set import TokenSet


class DefaultTextNormalizer():
    """
    Default Text Normalizer.

    Parameters
    ----------
    token_set: TokenSet
        TokenSet.
    """

    def __init__(self, token_set: TokenSet):
        
        self.token_set = token_set
        self.valid_tokens = self.token_set.non_special_tokens

        if self.token_set.letter_case == "lowercase":
            self.valid_tokens = set([x.lower() for x in self.valid_tokens])
        elif self.token_set.letter_case == "uppercase":
            self.valid_tokens = set([x.upper() for x in self.valid_tokens])

        self.invalid_chars_regex = f"[^\s{re.escape(''.join(set(self.valid_tokens)))}]"

    def __call__(self, sentence: str) -> str:
        """ 
        Preprocess the sentence for training/evaluation.

        Parameters:
        ----------
            sentence: str
                The sentence to be preprocessed.

        Returns:
        ----------
            str: The preprocessed sentence.
        """
        
        # to (lower|upper)case case
        if self.token_set.letter_case == "lowercase":
            sentence = sentence.lower()
        elif self.token_set.letter_case == "uppercase":
            sentence = sentence.upper()

        # removing invalid characters
        sentence = re.sub(self.invalid_chars_regex, " ", sentence)
        sentence = re.sub("\s+", " ", sentence).strip()
        
        return sentence
