from __future__ import annotations
import os
import sys
import torch
import logging
import numpy as np
from typing import Any, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from datasets import Dataset
from transformers import (
    Wav2Vec2Processor, 
    HfArgumentParser, 
    TrainingArguments as HFTrainingArguments, 
    set_seed,
    Trainer,
    AutoConfig,
    AutoModelForCTC,
    EarlyStoppingCallback
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.trainer_pt_utils import get_parameter_names
from huggingsound.metrics import cer, wer

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Model arguments

    Parameters:
    ----------
    freeze_feature_extractor: Optional[bool] = True
        Whether to freeze the feature extractor layers of the model.

    attention_dropout: Optional[float] = 0.1
        The dropout ratio for the attention probabilities.

    activation_dropout: Optional[float] = 0.1
        The dropout ratio for activations inside the fully connected layer.

    hidden_dropout: Optional[float] = 0.1
        The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.

    feat_proj_dropout: Optional[float] = 0.1
        The dropout probabilitiy for all 1D convolutional layers in feature extractor.

    final_dropout: Optional[float] = 0.1
        The dropout probabilitiy for CTC head.

    mask_time_prob: Optional[float] = 0.05
        Propability of each feature vector along the time axis to be chosen as the start of the vector
        span to be masked. Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature
        vectors will be masked along the time axis. This is only relevant if ``apply_spec_augment is True``.

    mask_time_length: Optional[int] = 10
        Length of vector span to mask along the time axis.

    mask_feature_prob: Optional[float] = 0.0
        Propability of each feature vector along the feature axis to be chosen as the start of the vector span to
        be masked. Approximately ``mask_time_prob * hidden_size // mask_time_length`` feature vectors will be
        masked along the time axis. This is only relevant if ``apply_spec_augment is True``."

    mask_feature_length: Optional[int] = 10
        Length of vector span to mask along the feature axis.

    layerdrop: Optional[float] = 0.0
        The probability to randomly drops layers at training time. For reference see https://arxiv.org/abs/1909.11556

    apply_spec_augment: Optional[bool] = True
        Whether to apply *SpecAugment* data augmentation to the outputs of the feature extractor. For reference see <https://arxiv.org/abs/1904.08779>"
    
    ctc_loss_reduction: Optional[str] = "sum"
        Specifies the reduction to apply to the output of ``torch.nn.CTCLoss``. Can be 'sum' or 'mean'.
    
    ctc_zero_infinity: Optional[bool] = False
        Whether to zero infinite losses and the associated gradients of ``torch.nn.CTCLoss``. Infinite losses
        mainly occur when the inputs are too short to be aligned to the targets
    """

    freeze_feature_extractor: bool = field(default=True)
    attention_dropout: float = field(default=0.1)
    activation_dropout: float = field(default=0.1)
    hidden_dropout: float = field(default=0.1)
    feat_proj_dropout: float = field(default=0.1)
    final_dropout: float = field(default=0.1)
    mask_time_prob: float = field(default=0.05)
    mask_time_length: int = field(default=10)
    mask_feature_prob: float = field(default=0.0)
    mask_feature_length: int = field(default=10)
    layerdrop: float = field(default=0.0)
    apply_spec_augment: bool = field(default=True)
    ctc_loss_reduction: str = field(default="sum")
    ctc_zero_infinity: bool = field(default=False)


@dataclass
class TrainingArguments:
    """
    Training arguments

    Parameters:
    ----------
    overwrite_output_dir: Optional[bool] = False
        If True, it will overwrite the content of the output directory. 
        If False, it will continue training if output directory points to a checkpoint directory.

    ignore_pretrained_weights: Optional[bool] = False
        If True, it will initializar all the model's weight randonly, i.e., it it will not use the pre-trained model weights during training.
    
    dataloader_num_workers: Optional[int] = 0
        Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the main process.
    
    learning_rate: Optional[float] = 3e-4
        The learning rate for AdamW optimizer.
    
    min_learning_rate: Optional[float] = 0.0
        The minimum learning rate for AdamW optimizer. (only used if lr_warmup_steps or lr_decay_steps are set)
    
    weight_decay: Optional[float] = 0.0
        The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in AdamW optimizer.
    
    adam_beta1: Optional[float] = 0.9
        The beta1 hyperparameter for the AdamW optimizer.
    
    adam_beta2: Optional[float] = 0.999
        The beta2 hyperparameter for the AdamW optimizer.
    
    adam_epsilon: Optional[float] = 1e-8
        The epsilon hyperparameter for the AdamW optimizer.
    
    max_grad_norm: Optional[float] = 1.0
        Maximum gradient norm (for gradient clipping).
    
    lr_warmup_steps: Optional[int] = 0
        Number of warmup steps for learning rate scheduler at the beginning of training.
    
    lr_decay_steps: Optional[int] = 0
        Number of decay steps for learning rate scheduler at the end of training.
    
    eval_steps: Optional[int] = None
        Run an evaluation every X steps.
    
    group_by_length: Optional[bool] = True
        Whether or not to group together samples of roughly the same length in the training dataset (to minimize padding applied and be more efficient).
    
    gradient_accumulation_steps: Optional[int] = 1
        Number of updates steps to accumulate before performing a backward/update pass.

    gradient_checkpointing: Optional[bool] = True
        If True, use gradient checkpointing to save memory at the expense of slower backward pass.
    
    pad_to_multiple_of: Optional[int] = None
        If set will pad the sequence to a multiple of the provided value.
        This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).

    per_device_train_batch_size: Optional[int] = 8
        The batch size per GPU/TPU core/CPU for training.
    
    per_device_eval_batch_size: Optional[int] = 8
        The batch size per GPU/TPU core/CPU for evaluation.
    
    fp16: Optional[bool] = False
        Whether to use 16-bit (mixed) precision training instead of 32-bit training.

    use_8bit_optimizer: Optional[bool] = False
        Whether to use 8-bit optimizer.
    
    logging_steps: Optional[int] = 100
        Log every X updates steps.
    
    num_train_epochs: Optional[float] = 3.0
        Total number of training epochs to perform (if not an integer, will perform the decimal part percents of
        the last epoch before stopping training).
    
    max_steps: Optional[int] = 0
        If set to a positive number, the total number of training steps to perform. Overrides num_train_epochs.
    
    report_to: Optional[list[str]] = ["none"]
        The list of integrations to report the results and logs to. Supported platforms are "azure_ml",
        "comet_ml", "mlflow", "tensorboard" and "wandb". Use "all" to report to
        all integrations installed, "none" for no integrations.
    
    save_total_limit: Optional[int] = None
        Limit the total amount of checkpoints. Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints
    
    metric_for_best_model: Optional[str] = "cer"
        The metric to use to compare two different models.

    _n_gpu: Optional[int] = 1
        Numbers of gpus to use for training.
    
    seed: Optional[int] = None
        Random seed that will be set at the beginning of training. To ensure reproducibility across runs.

    training_step_callbacks: list[Callable[transformers.Trainer, float]] = []
        A list of callbacks that will be called at the end of each training step.
        The trainer, model and the current loss will be passed to those functions: f(trainer: transformers.Trainer, loss: float)

    batch_creation_callbacks: list[Callable[ list[dict[str, list[int]]] ]] = []
        A list of callbacks that will be called at the end of each batch creation.
        The batch will be passed to those functions: f(batch: list[dict[str, list[int]]])

    evaluation_callbacks: list[Callable[dict, list, list]] = []
        A list of callbacks that will be called at the end of each evaluation step.
        The metrics (WER, CER), references and predictions will be passed to those functions: f(metrics: dict, references: list, predictions: list)

    metrics_batch_size: Optional[int] = None
        Batch size to use for evaluation. When this value is specified, the evaluation function will chunk the data into 
        batches of the specified size and compute the metrics on each batch.
        After all batches are computed, the function will compute the average metrics over all batches.
        (You will probably need to define this if you have memory issues).
    
    show_dataset_stats: Optional[bool] = False
        Whether to show dataset stats

    early_stopping_patience: Optional[int] = None
        Use with metric_for_best_model to stop training when the specified metric worsens for early_stopping_patience evaluation calls.

    load_best_model_at_end: Optional[bool] = False
        Whether to load the best model at the end of training.

    """
    
    overwrite_output_dir: bool = field(default=False)
    ignore_pretrained_weights: bool = field(default=False)
    dataloader_num_workers: int = field(default=0)
    learning_rate: float = field(default=3e-4)
    min_learning_rate: float = field(default=0.0)
    weight_decay: float = field(default=0.0)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)
    max_grad_norm: float = field(default=1.0)
    lr_warmup_steps: int = field(default=0)
    lr_decay_steps: int = field(default=0)
    eval_steps: int = field(default=None)
    group_by_length: bool = field(default=True)
    length_column_name: str = field(default="length")
    gradient_accumulation_steps: int = field(default=1)
    gradient_checkpointing: bool = field(default=True)
    pad_to_multiple_of: int = field(default=None)
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=8)
    fp16: bool = field(default=False)
    use_8bit_optimizer: bool = field(default=False)
    logging_steps: int = field(default=100)
    num_train_epochs: float = field(default=3.0)
    max_steps: int = field(default=0)
    report_to: list[str] = field(default_factory=lambda: ["none"])
    save_total_limit: int = field(default=None)
    metric_for_best_model: str = field(default=None)
    _n_gpu: int = field(default=1)
    seed: int = field(default=42)
    training_step_callbacks: list[Callable] = field(default_factory=lambda: [])
    batch_creation_callbacks: list[Callable] = field(default_factory=lambda: [])
    evaluation_callbacks: list[Callable] = field(default_factory=lambda: [])
    metrics_batch_size: int = field(default=None)
    show_dataset_stats: bool = field(default=True)
    early_stopping_patience: int = field(default=None)
    load_best_model_at_end: bool = field(default=False)

class CTCDataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.

    Parameters:
    ----------
        processor: Wav2Vec2Processor
            The processor used for proccessing the data.

        batch_creation_callbacks: Optional[list[dict[str, Union[list[int], torch.Tensor]]]] = None
            A list of callbacks that will be called at the end of each batch creation.

        pad_to_multiple_of: Optional[int] = None
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
    """

    def __init__(self, processor: Wav2Vec2Processor, batch_creation_callbacks: Optional[list[Callable]] = None, 
                 pad_to_multiple_of: Optional[int] = None):
        self.processor = processor
        self.batch_creation_callbacks = batch_creation_callbacks
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Union[list[int], torch.Tensor]]]) -> dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods

        if self.batch_creation_callbacks is not None:
            for callback in self.batch_creation_callbacks:
                callback(features)

        input_features = [{"input_values": feature["input_values"]} for feature in features]
        labels = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features=input_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=labels,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # replace padding with -100 to ignore loss correctly
        batch["labels"] = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        return batch


class CTCTrainer(Trainer):

    """
    Data collator that will dynamically pad the inputs received.

    Parameters:
    ----------
        min_learning_rate: Optional[float] = None
            The minimum learning rate to use.

        lr_warmup_steps: Optional[int] = None
            The number of steps to use for a warmup.

        lr_decay_steps: Optional[int] = None
            The number of steps to use for a decay.
            
        training_step_callbacks: list[Callable[transformers.Trainer, float]] = []
            A list of callbacks that will be called at the end of each training step.
            The trainer, model and the current loss will be passed to those functions: f(trainer: transformers.Trainer, loss: float)
    """

    def __init__(self, min_learning_rate=None, lr_warmup_steps=None, lr_decay_steps=None, 
                 training_step_callbacks: Optional[list[Callable]] = None, use_8bit_optimizer: Optional[bool] = False, **kwargs):

        super().__init__(**kwargs)
        self.min_learning_rate = min_learning_rate
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_decay_steps = lr_decay_steps
        self.training_step_callbacks = training_step_callbacks
        self.use_8bit_optimizer = use_8bit_optimizer

    def create_optimizer(self):

        decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]

        optimizer_class = torch.optim.AdamW

        if self.use_8bit_optimizer:
            
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError("To use 8bit optimizer please install the bitsandbytes from https://github.com/facebookresearch/bitsandbytes")

            optimizer_class = bnb.optim.AdamW

        self.optimizer = optimizer_class(
            params=optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )

        return self.optimizer
    
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. This scheduler supports a warmup phase where the learning rate linearly increases 
        at the beginning of the training. And a decay phase where the learning rate decreases over time at the end of the training.
        
        Parameters:
        ----------
            num_training_steps (int): The number of total training steps to do.
        """

        min_lr_ratio = self.min_learning_rate / self.args.learning_rate

        def _lr_lambda(current_step, num_training_steps=num_training_steps, lr_decay_steps=self.lr_decay_steps,
                       lr_warmup_steps=self.lr_warmup_steps, min_lr_ratio=min_lr_ratio):

            if lr_warmup_steps is not None and current_step < lr_warmup_steps: # warmup phase
                return max(
                    min_lr_ratio, float(current_step) / float(max(1, lr_warmup_steps))
                )
            elif lr_decay_steps is not None and current_step >= num_training_steps - lr_decay_steps: # decay phase
                return max(
                    min_lr_ratio, float(num_training_steps - current_step) / float(max(1, lr_decay_steps))
                )
            else: # constant learning rate phase
                return 1
        
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, _lr_lambda)


    def training_step(self, model: torch.nn.Module, inputs: dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Parameters:
        ----------
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Returns:
        ----------
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)

        if (hasattr(self, 'use_amp') and self.use_amp) or (hasattr(self, 'use_cuda_amp') and self.use_cuda_amp):
            with torch.cuda.amp.autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            if model.module.config.ctc_loss_reduction == "mean":
                loss = loss.mean()
            elif model.module.config.ctc_loss_reduction == "sum":
                loss = loss.sum() / (inputs["labels"] >= 0).sum()
            else:
                raise ValueError(f"{model.config.ctc_loss_reduction} is not valid. Choose one of ['mean', 'sum']")

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if (hasattr(self, 'use_amp') and self.use_amp) or (hasattr(self, 'use_cuda_amp') and self.use_cuda_amp):
            self.scaler.scale(loss).backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        if self.training_step_callbacks is not None:
            for callback in self.training_step_callbacks:
                callback(self, float(loss.detach()))

        return loss.detach()


def finetune_ctc(model_name_or_path: str, output_dir: str, processor: Wav2Vec2Processor, train_dataset: Dataset, 
                 eval_dataset: Optional[Dataset] = None, device: Optional[str] = "cpu", 
                 training_args: Optional[TrainingArguments] = None, model_args: Optional[ModelArguments] = None):
    """
    Finetunes the CTC model.

    Parameters:
    ----------
    
    model_name_or_path : str
        The path to the model or the model identifier from huggingface.co/models.

    output_dir : str
        The output directory where the model checkpoints will be written.

    processor : transformers.Wav2Vec2Processor
        The audio processor object.

    train_dataset : transformers.Dataset
        The training dataset.
    
    eval_dataset : Optional[transformers.Dataset] = None
        The evaluation dataset.
    
    device : str = "cpu"
        The device to run the model training on.

    training_args : Optional[TrainingArguments] = None
        The training arguments.

    model_args : Optional[ModelArguments] = None
        The model arguments.
        
    """

    if training_args is None:
        training_args = TrainingArguments()

    if model_args is None:
        model_args = ModelArguments()

    hf_arg_parser = HfArgumentParser((HFTrainingArguments))
    training_args_dict = asdict(training_args)
    training_args_dict["output_dir"] = output_dir
    training_args_dict["device"] = device
    training_args_dict["no_cuda"] = device == "cpu"
    training_args_dict["do_train"] = True

    if eval_dataset is not None:
        training_args_dict["do_eval"] = True
        training_args_dict["evaluation_strategy"] = "steps"
        training_args_dict["greater_is_better"] = False
    
    hftraining_args = hf_arg_parser.parse_dict(training_args_dict, allow_extra_keys=True)[0]

    # Set seed before initializing model
    if hftraining_args.seed is not None:
        set_seed(hftraining_args.seed)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(hftraining_args.output_dir) and not hftraining_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(hftraining_args.output_dir)
        if last_checkpoint is None and len(os.listdir(hftraining_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({hftraining_args.output_dir}) already exists and is not empty. "
                "Clear the folder or set TrainingArguments.overwrite_output_dir to True to overcome that."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the model's output directory or set TrainingArguments.overwrite_output_dir to False to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(hftraining_args.local_rank) else logging.WARN)

    if training_args.show_dataset_stats:
        logger.info("Getting dataset stats...")

        total_train_size_in_seconds = 0
        for sample in train_dataset:
            total_train_size_in_seconds += len(sample["input_values"]) / processor.feature_extractor.sampling_rate
        logger.info(f"Training dataset size: {len(train_dataset)} samples, {total_train_size_in_seconds/3600} hours")

        if eval_dataset is not None:
            total_eval_size_in_seconds = 0
            for sample in eval_dataset:
                total_eval_size_in_seconds += len(sample["input_values"]) / processor.feature_extractor.sampling_rate
            logger.info(f"Evaluation dataset size: {len(eval_dataset)} samples, {total_eval_size_in_seconds/3600} hours")

    # Setup model

    config = AutoConfig.from_pretrained(model_name_or_path)

    config.update(
        {
            "feat_proj_dropout": model_args.feat_proj_dropout,
            "attention_dropout": model_args.attention_dropout,
            "hidden_dropout": model_args.hidden_dropout,
            "final_dropout": model_args.final_dropout,
            "mask_time_prob": model_args.mask_time_prob,
            "mask_time_length": model_args.mask_time_length,
            "mask_feature_prob": model_args.mask_feature_prob,
            "mask_feature_length": model_args.mask_feature_length,
            "layerdrop": model_args.layerdrop,
            "activation_dropout": model_args.activation_dropout,
            "ctc_loss_reduction": model_args.ctc_loss_reduction,
            "pad_token_id": processor.tokenizer.pad_token_id,
            "vocab_size": len(processor.tokenizer),
        }
    )

    if training_args.ignore_pretrained_weights:
        model = AutoModelForCTC.from_config(config=config)
    else:
        model = AutoModelForCTC.from_pretrained(model_name_or_path, config=config)

    if hftraining_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if model_args.freeze_feature_extractor:    
        model.freeze_feature_extractor()

    # Defining metrics
    def _compute_metrics(pred, processor=processor, metrics_batch_size=training_args.metrics_batch_size, 
                         evaluation_callbacks=training_args.evaluation_callbacks):

        pred_logits = pred.predictions
        padding_mask = pred_logits[:,:,0] == -100
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred_ids[padding_mask] = processor.tokenizer.pad_token_id

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        metrics = {
            "wer": wer(predictions=pred_str, references=label_str, chunk_size=metrics_batch_size),
            "cer": cer(predictions=pred_str, references=label_str, chunk_size=metrics_batch_size),
        }

        if evaluation_callbacks is not None:
            for callback in evaluation_callbacks:
                callback(metrics, label_str, pred_str)

        return metrics

    # Setup data collator

    data_collator = CTCDataCollatorWithPadding(
        processor=processor,
        batch_creation_callbacks=training_args.batch_creation_callbacks,
        pad_to_multiple_of=training_args.pad_to_multiple_of,
    )
    
    # Initialize Trainer

    logger.info("Building trainer...")

    callbacks = []
    if training_args.early_stopping_patience is not None and training_args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience)
        )
        hftraining_args.load_best_model_at_end = True

    trainer = CTCTrainer(
        args=hftraining_args,
        model = model,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=_compute_metrics if eval_dataset is not None else None,
        min_learning_rate=training_args.min_learning_rate,
        lr_warmup_steps=training_args.lr_warmup_steps,
        lr_decay_steps=training_args.lr_decay_steps,
        training_step_callbacks=training_args.training_step_callbacks,
        use_8bit_optimizer=training_args.use_8bit_optimizer,
        callbacks = callbacks
    )

    # Training

    logger.info("Starting training...")

    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    elif os.path.isdir(model_name_or_path):
        checkpoint = model_name_or_path
    else:
        checkpoint = None

    # Saving processor config to output directory

    if is_main_process(hftraining_args.local_rank):
        processor.save_pretrained(hftraining_args.output_dir)
    
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()

    metrics = train_result.metrics  
    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
