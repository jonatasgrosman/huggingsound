from __future__ import annotations
import os
import sys
import torch
import warnings
import logging
from typing import Optional, Callable
from datasets import load_from_disk, Dataset
from tqdm import tqdm
from transformers import (
    Wav2Vec2Processor, 
    AutoConfig,
    AutoModelForCTC
)
from transformers.models.auto.modeling_auto import MODEL_FOR_CTC_MAPPING_NAMES
from huggingsound.utils import get_chunks, get_waveforms, get_dataset_from_dict_list
from huggingsound.token_set import TokenSet
from huggingsound.normalizer import DefaultTextNormalizer
from huggingsound.trainer import TrainingArguments, ModelArguments, finetune_ctc
from huggingsound.speech_recognition.decoder import Decoder, GreedyDecoder
from huggingsound.metrics import cer, wer

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)

class SpeechRecognitionModel():
    """
    Speech Recognition Model.

    Parameters
    ----------
    model_path : str
        The path to the model or the model identifier from huggingface.co/models.
    
    device: Optional[str] = "cpu"
        Device to use for inference/evaluation/training, default is "cpu". If you want to use a GPU for that, 
        you'll probably need to specify the device as "cuda"
        
    letter_case: Optional[str] = "lowercase"
        Letter case to use for the transcription, can be "lowercase", "uppercase" or None. 
        If None, the transcription will be in the same case as the model's output.
    """

    def __init__(self, model_path: str, device: Optional[str] = "cpu", letter_case: Optional[str] = "lowercase"):
        
        self.model_path = model_path
        self.device = device
        self.letter_case = letter_case # TODO: may this be useful?
        
        logger.info("Loading model...")
        self._load_model()

    def _load_model(self):

        self.model_config = AutoConfig.from_pretrained(self.model_path)

        ctc_finetuded_architectures = set(MODEL_FOR_CTC_MAPPING_NAMES.values())

        self.is_finetuned = len(ctc_finetuded_architectures.intersection(self.model_config.architectures)) > 0

        if not self.is_finetuned:

            logger.warning("Not fine-tuned model! You'll need to fine-tune it before use this model for audio transcription")

        else:

            self.processor = Wav2Vec2Processor.from_pretrained(self.model_path)
            self.model = AutoModelForCTC.from_pretrained(self.model_path)
            self.model.to(self.device)

            self.token_set = TokenSet.from_processor(self.processor)


    def transcribe(self, paths: list[str], batch_size: Optional[int] = 1, decoder: Optional[Decoder] = None) -> list[dict]:
        """ 
        Transcribe audio files.

        Parameters:
        ----------
            paths: list[str]
                List of paths to audio files to transcribe

            batch_size: Optional[int] = 1
                Batch size to use for inference

            decoder: Optional[Decoder] = None
                Decoder to use for transcription. If you don't specify this, the engine will use the GreedyDecoder.

        Returns:
        ----------
            list[dict]:
                A list of dictionaries containing the transcription for each audio file:

                [{
                    "transcription": str,
                    "start_timesteps": list[int],
                    "end_timesteps": list[int],
                    "probabilities": list[float]
                }, ...]
        """

        if not self.is_finetuned:
            raise ValueError("Not fine-tuned model! Please, fine-tune the model first.")
        
        if decoder is None:
            decoder = GreedyDecoder(self.token_set)

        sampling_rate = self.processor.feature_extractor.sampling_rate
        result = []

        for paths_batch in tqdm(list(get_chunks(paths, batch_size))):

            waveforms = get_waveforms(paths_batch, sampling_rate)

            inputs = self.processor(waveforms, sampling_rate=sampling_rate, return_tensors="pt", padding=True, do_normalize=True)

            with torch.no_grad():
                logits = self.model(inputs.input_values.to(self.device), attention_mask=inputs.attention_mask.to(self.device)).logits

            result += decoder(logits)

        return result 

    def evaluate(self, references: list[dict], predictions: Optional[list[dict]] = None, metrics_batch_size: Optional[int] = None, 
                 inference_batch_size: Optional[int] = 1, decoder: Optional[Decoder] = None, text_normalizer: Callable[[str], str] = None) -> dict:
        """ 
        Evaluate the model.

        Parameters:
        ----------
            references: list[dict]
                List of dictionaries containing the reference transcriptions for each audio file.
                The dictionaries should have the following structure:

                [{
                    "transcription": str,
                    "path": str,
                }, ...]

            predictions: Optional[list[dict]] = None
                List of dictionaries containing the predictions for each audio file.
                If this list is not provided, the engine will execute the transcribe() using the references.
                The dictionaries should have the following structure:

                [{
                    "transcription": str,
                }, ...]

            metrics_batch_size: Optional[int] = None
                Batch size to use for evaluation. When this value is specified, the evaluation function will chunk the data into 
                batches of the specified size and compute the metrics on each batch.
                After all batches are computed, the function will compute the average metrics over all batches.
                (You will probably need to define this if you have memory issues).

            inference_batch_size: Optional[int] = 1
                Batch size to use for inference.

            decoder: Optional[Decoder] = None
                Decoder to use for transcription. If you don't specify this, the engine will use the GreedyDecoder.
            
            text_normalizer: Callable[[str], str] = None
                Function used to normalize the transcriptions before evaluation.

        Returns:
        ----------
            dict:
                A dictionary containing the evaluation metrics:

                {
                    "cer": float,
                    "wer": float,
                }
        """

        if not self.is_finetuned:
            raise ValueError("Not fine-tuned model! Please, fine-tune the model first.")

        if decoder is None:
            decoder = GreedyDecoder(self.token_set)

        if text_normalizer is None:
            text_normalizer = DefaultTextNormalizer(self.token_set)

        if predictions is None:
            paths = [x["path"] for x in references]
            predictions = self.transcribe(paths, inference_batch_size, decoder)
        
        evaluation = {}
        reference_transcriptions = [text_normalizer(x["transcription"]) for x in references]
        predicted_transcriptions = [text_normalizer(x["transcription"]) for x in predictions]

        evaluation = {
            "cer": cer(predictions=predicted_transcriptions, references=reference_transcriptions, chunk_size=metrics_batch_size),
            "wer": wer(predictions=predicted_transcriptions, references=reference_transcriptions, chunk_size=metrics_batch_size)
        }

        return evaluation

    def _prepare_dataset_for_finetuning(self, dataset: Dataset, processor: Wav2Vec2Processor, text_normalizer: Callable[[str], str],
                                        length_column_name: str, num_workers: int) -> Dataset:
    
        def __process_dataset_sample(sample, text_normalizer=text_normalizer, 
                                    processor=processor, length_column_name=length_column_name):

            # Build input values
            sampling_rate = processor.feature_extractor.sampling_rate
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    waveform = get_waveforms([sample["path"]], sampling_rate)[0]
            except Exception as e:
                logger.error(f"Loading error for {sample['path']}: {e}")
                raise e
            sample["input_values"] = processor(waveform, sampling_rate=sampling_rate, do_normalize=True).input_values[0]

            # Building labels
            ## appending a " " because the CTC loss concatenates all batches into a single vector, so we need to separate sentences by a whitespace
            transcription = text_normalizer(sample["transcription"]) + " " 
            with processor.as_target_processor():
                sample["labels"] = processor(transcription).input_ids

            # Building length
            sample[length_column_name] = len(sample["input_values"])
            
            return sample

        dataset = dataset.map(
            __process_dataset_sample,
            remove_columns=dataset.column_names,
            num_proc=num_workers
        )

        return dataset

    def _get_dataset(self, processor: Wav2Vec2Processor, text_normalizer: Callable[[str], str], data: Optional[list[dict]] = None, 
                     data_cache_dir: Optional[str] = None, length_column_name: Optional[str] = "length", 
                     num_workers: Optional[int] = None) -> Dataset:

        assert data is not None or data_cache_dir is not None, "at least one of data parameters (data or data_cache_dir) needs to be specified"

        if data_cache_dir is not None and os.path.isfile(os.path.join(data_cache_dir, "dataset_info.json")):
            logger.info("Loading data from cache...")
            dataset = load_from_disk(data_cache_dir)
        else:
            logger.info("Converting data format...")
            dataset = get_dataset_from_dict_list(data)
            logger.info("Preparing data input and labels...")
            dataset = self._prepare_dataset_for_finetuning(dataset, processor, text_normalizer, length_column_name, num_workers)

            if data_cache_dir is not None:
                logger.info("Caching data...")
                dataset.save_to_disk(data_cache_dir)

        return dataset

    def finetune(self, output_dir: str, train_data: list[dict] = None, eval_data: Optional[list[dict]] = None, 
                 data_cache_dir: Optional[str] = None, token_set: Optional[TokenSet] = None, 
                 training_args: Optional[TrainingArguments] = None, model_args: Optional[ModelArguments] = None, 
                 text_normalizer: Callable[[str], str] = None, num_workers: Optional[int] = 1):
        """
        Finetune the model.

        Parameters
        ----------
        output_dir: str
            The output directory where the model checkpoints will be written.
        
        train_data: list[dict] = None
            A list of dict in the format {path: str, transcription: str} for training. 
            This parameter is optional only if data_cache_dir is specified and filled with already preprocessed data.
        
        eval_data: Optional[list[dict]] = None
            A list of dict in the format {path: str, transcription: str} for evaluation
        
        data_cache_dir: Optional[str] = None
            Pre-processed dataset cache directory. This can decrease the time needed to start the training (by using a lot of disk space).
        
        token_set: Optional[TokenSet] = None
            The token set to be used for training. This is mandatory if the model is not already fine-tuned.

        training_args: Optional[TrainingArguments] = None
            The training arguments.

        model_args: Optional[ModelArguments] = None
            The model arguments.

        text_normalizer: Callable[[str], str] = None
            Function used to normalize the transcriptions before evaluation.
        
        num_workers: Optional[int] = 1
            Number of workers to use for data loading.
        """

        if train_data is None and data_cache_dir is None:
            raise ValueError("train_data or data_cache_dir must be specified")

        if not self.is_finetuned and token_set is None:
            raise ValueError("The model is not fine-tuned yet, so you need to provide a token_set to fine-tune it")
        
        if self.is_finetuned:
            if token_set is not None:
                logger.warning("The model is already fine-tuned. So the provided token_set won't be used. The model's token_set will be used instead")
            token_set = self.token_set

        if text_normalizer is None:
            text_normalizer = DefaultTextNormalizer(token_set)

        if training_args is None:
            training_args = TrainingArguments()
        
        if model_args is None:
            model_args = ModelArguments()

        processor = self.processor if self.is_finetuned else token_set.to_processor(self.model_path)

        os.makedirs(output_dir, exist_ok=True)

        train_data_cache_dir = None
        eval_data_cache_dir = None

        if data_cache_dir is not None:
            train_data_cache_dir = os.path.join(data_cache_dir, "train")
            os.makedirs(train_data_cache_dir, exist_ok=True)

            eval_data_cache_dir = os.path.join(data_cache_dir, "eval")
            os.makedirs(eval_data_cache_dir, exist_ok=True)

        logger.info("Loading training data...")
        train_dataset = self._get_dataset(processor, text_normalizer, train_data, train_data_cache_dir, training_args.length_column_name, num_workers)
        
        eval_dataset = None
        if eval_data is not None or data_cache_dir is not None:
            logger.info("Loading evaluation data...")
            eval_dataset = self._get_dataset(processor, text_normalizer, eval_data, eval_data_cache_dir, training_args.length_column_name, num_workers)
        
        logger.info("Starting fine-tuning process...")
        
        finetune_ctc(self.model_path, output_dir, processor, train_dataset, eval_dataset, self.device, training_args, model_args)

        logger.info("Loading fine-tuned model...")

        self.model_path = output_dir
        self._load_model()
