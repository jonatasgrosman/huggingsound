from __future__ import annotations
from typing import Optional
import jiwer
import gc

def wer(predictions: list[str], references: list[str], chunk_size: Optional[int] = None) -> float:
    """
    Computes WER score of transcribed segments against references.

    Word error rate (WER) is a common metric of the performance of an automatic speech recognition system.
    The general difficulty of measuring performance lies in the fact that the recognized word sequence can 
    have a different length from the reference word sequence (supposedly the correct one). 
    The WER is derived from the Levenshtein distance, working at the word level instead of the phoneme level.

    Word error rate can then be computed as:
    WER = (S + D + I) / N = (S + D + I) / (S + D + C)
    where
    S is the number of substitutions,
    D is the number of deletions,
    I is the number of insertions,
    C is the number of correct words,
    N is the number of words in the reference (N=S+D+C).

    WER's output is always a number between 0 and 1. This value indicates the percentage of words that were incorrectly predicted. 
    The lower the value, the better the performance of the ASR system with a WER of 0 being a perfect score.
    
    Parameters:
    ----------
        predictions: list[str]
            List of transcribtions to score.
        
        references: list[str]
            List of references for each speech input.
        
        chunk_size: Optional[int] = None
            Size of the chunk to use for computation. 
            When this value is specified, the function will chunk the data into batches of the specified size and compute the WER on each batch.
            After all batches are computed, the function will compute the average WER over all batches.
            (You will probably need to define this if you have memory issues).
    
    Returns:
    ----------
        float: the word error rate
    """
    
    if chunk_size is None: return jiwer.wer(references, predictions)
    
    start = 0
    end = chunk_size
    H, S, D, I = 0, 0, 0, 0

    while start < len(references):

        chunk_metrics = jiwer.compute_measures(references[start:end], predictions[start:end], 
                                               truth_transform=jiwer.transformations.wer_default, 
                                               hypothesis_transform=jiwer.transformations.wer_default)
        H = H + chunk_metrics["hits"]
        S = S + chunk_metrics["substitutions"]
        D = D + chunk_metrics["deletions"]
        I = I + chunk_metrics["insertions"]
        start += chunk_size
        end += chunk_size

        # sometimes this metric uses a lot of memory, so we'll try to free it here
        del chunk_metrics
        gc.collect()
    
    return float(S + D + I) / float(H + S + D)


def cer(predictions: list[str], references: list[str], chunk_size: Optional[int] = None) -> float:
    """
    Computes CER score of transcribed segments against references.

    Character error rate (CER) is a common metric of the performance of an automatic speech recognition system.
    CER is similar to Word Error Rate (WER), but operate on character insted of word. Please refer to docs of WER for further information.
    
    Character error rate can be computed as:
    CER = (S + D + I) / N = (S + D + I) / (S + D + C)
    where
    S is the number of substitutions,
    D is the number of deletions,
    I is the number of insertions,
    C is the number of correct characters,
    N is the number of characters in the reference (N=S+D+C).

    CER's output is always a number between 0 and 1. This value indicates the percentage of characters that were incorrectly predicted. 
    The lower the value, the better the performance of the ASR system with a CER of 0 being a perfect score.
    
    Parameters:
    ----------
        predictions: list[str]
            List of transcribtions to score.
        
        references: list[str]
            List of references for each speech input.
        
        chunk_size: Optional[int] = None
            Size of the chunk to use for computation. 
            When this value is specified, the function will chunk the data into batches of the specified size and compute the CER on each batch.
            After all batches are computed, the function will compute the average CER over all batches.
            (You will probably need to define this if you have memory issues).
   
    Returns:
    ----------
        float: the character error rate
    """

    if chunk_size is None: return jiwer.cer(references, predictions)

    start = 0
    end = chunk_size
    H, S, D, I = 0, 0, 0, 0

    while start < len(references):
        chunk_metrics = jiwer.cer(references[start:end], predictions[start:end], return_dict=True)
        H = H + chunk_metrics["hits"]
        S = S + chunk_metrics["substitutions"]
        D = D + chunk_metrics["deletions"]
        I = I + chunk_metrics["insertions"]
        start += chunk_size
        end += chunk_size

        # sometimes this metric uses a lot of memory, so we'll try to free it here
        del chunk_metrics
        gc.collect()
    
    return float(S + D + I) / float(H + S + D)
