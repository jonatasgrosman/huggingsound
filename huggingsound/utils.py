from __future__ import annotations
import warnings
import librosa
import numpy as np
from typing import Iterator, Optional
from datasets import Dataset
from collections import Counter


def get_chunks(values: list, n: int) -> Iterator:
    """ 
    Yield successive n-sized chunks from values.

    Parameters:
    ----------
        values: list
            values to be chunked
       
        n: int
            chunk size

    Returns:
    ----------
        Iterator: A chunk iterator
    """

    for i in range(0, len(values), n):
        yield values[i:i + n]


def get_waveforms(pahts: list[str], sampling_rate: Optional[int] = 16000) -> list[np.ndarray]:
    """ 
    Get waveforms from audio files.

    Parameters:
    ----------
        pahts: list[str]
            paths to audio files
        
        sampling_rate: Optional[int] = 16000
            sampling rate of waveforms

    Returns:
    ----------
        list[np.ndarray]: waveforms from audio files
    """

    waveforms = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for path in pahts:
            waveform, sr = librosa.load(path, sr=sampling_rate)
            waveforms.append(waveform)
    
    return waveforms


def get_dataset_from_dict_list(data: list[dict]) -> Dataset:
    """ 
    Get dataset from dict list.

    Parameters:
    ----------
        data: list[dict]
            dict list to be converted to dataset

    Returns:
    ----------
        Dataset: dataset from dict list
    """

    keys = data[0].keys()
    transformed_data = {}
    for key in keys:
        for d in data:
            transformed_data[key] = transformed_data.get(key, []) + [d[key]]

    dataset = Dataset.from_dict(transformed_data)

    return dataset


class CustomCounter(Counter):
    """ The operation 'Counter() + Counter()' ignore keys with zero values, this Counter extension fix that"""
    
    def __add__(self, other):
        if not isinstance(other, Counter):
            return NotImplemented
        result = CustomCounter()
        for elem, count in self.items():
            newcount = count + other[elem]
            result[elem] = newcount
        for elem, count in other.items():
            if elem not in self:
                result[elem] = count
        return result