# HuggingSound

HuggingSound: A toolkit for speech-related tasks based on [HuggingFace's](https://huggingface.co/) tools.

I have no intention of building a very complex tool here. 
I just wanna have an easy-to-use toolkit for my speech-related experiments.
I hope this library could be helpful for someone else too :)

# Requirements

- Python 3.7+

# Installation

```console
$ pip install huggingsound
```

# How to use it?

I'll try to summarize the usage of this toolkit. 
But many things will be missing from the documentation below. I promise to make it better soon.
For now, you can open an issue if you have some questions or look at the source code to see how it works.
You can check more usage examples in the repository `examples` folder.

## Speech recognition

For speech recognition you can use any CTC model hosted on the Hugging Face Hub. You can find some available models [here](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition).

### Inference

```python
from huggingsound import SpeechRecognitionModel

model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english")
audio_paths = ["/path/to/sagan.mp3", "/path/to/asimov.wav"]

transcriptions = model.transcribe(audio_paths)

print(transcriptions)

# transcriptions format (a list of dicts, one for each audio file):
# [
#  {
#   "transcription": "extraordinary claims require extraordinary evidence", 
#   "start_timestamps": [100, 120, 140, 180, ...],
#   "end_timestamps": [120, 140, 180, 200, ...],
#   "probabilities": [0.95, 0.88, 0.9, 0.97, ...]
# },
# ...]
#
# as you can see, not only the transcription is returned but also the timestamps (in milliseconds) 
# and probabilities of each character of the transcription.

```

### Inference (boosted by a language model)

```python
from huggingsound import SpeechRecognitionModel, KenshoLMDecoder

model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english")
audio_paths = ["/path/to/sagan.mp3", "/path/to/asimov.wav"]

# The LM format used by the LM decoders is the KenLM format (arpa or binary file).
# You can download some LM files examples from here: https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english/tree/main/language_model
lm_path = "path/to/your/lm_files/lm.binary"
unigrams_path = "path/to/your/lm_files/unigrams.txt"

# We implemented three different decoders for LM boosted decoding: KenshoLMDecoder, ParlanceLMDecoder, and FlashlightLMDecoder
# On this example, we'll use the KenshoLMDecoder
# To use this decoder you'll need to install the Kensho's ctcdecode first (https://github.com/kensho-technologies/pyctcdecode)
decoder = KenshoLMDecoder(model.token_set, lm_path=lm_path, unigrams_path=unigrams_path)

transcriptions = model.transcribe(audio_paths, decoder=decoder)

print(transcriptions)

```

### Evaluation
```python
from huggingsound import SpeechRecognitionModel

model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english")

references = [
    {"path": "/path/to/sagan.mp3", "transcription": "extraordinary claims require extraordinary evidence"},
    {"path": "/path/to/asimov.wav", "transcription": "violence is the last refuge of the incompetent"},
]

evaluation = model.evaluate(references)

print(evaluation)

# evaluation format: {"wer": 0.08, "cer": 0.02}
```

### Fine-tuning
```python
from huggingsound import TrainingArguments, ModelArguments, SpeechRecognitionModel, TokenSet

model = SpeechRecognitionModel("facebook/wav2vec2-large-xlsr-53")
output_dir = "my/finetuned/model/output/dir"

# first of all, you need to define your model's token set
# however, the token set is only needed for non-finetuned models
# if you pass a new token set for an already finetuned model, it'll be ignored during training
tokens = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]
token_set = TokenSet(tokens)

# define your train/eval data
train_data = [
    {"path": "/path/to/sagan.mp3", "transcription": "extraordinary claims require extraordinary evidence"},
    {"path": "/path/to/asimov.wav", "transcription": "violence is the last refuge of the incompetent"},
]
eval_data = [
    {"path": "/path/to/sagan2.mp3", "transcription": "absence of evidence is not evidence of absence"},
    {"path": "/path/to/asimov2.wav", "transcription": "the true delight is in the finding out rather than in the knowing"},
]

# and finally, fine-tune your model
model.finetune(
    output_dir, 
    train_data=train_data, 
    eval_data=eval_data, # the eval_data is optional
    token_set=token_set,
)

```

# Troubleshooting

- If you are having trouble when loading MP3 files: `$ sudo apt-get install ffmpeg`

# Want to help?

See the [contribution guidelines](https://github.com/jonatasgrosman/huggingsound/blob/master/CONTRIBUTING.md)
if you'd like to contribute to HuggingSound project.

You don't even need to know how to code to contribute to the project. Even the improvement of our documentation is an outstanding contribution.

If this project has been useful for you, please share it with your friends. This project could be helpful for them too.

If you like this project and want to motivate the maintainers, give us a :star:. This kind of recognition will make us very happy with the work that we've done with :heart:

You can also [sponsor me](https://github.com/sponsors/jonatasgrosman) :heart_eyes:

# Citation
If you want to cite the tool you can use this:

```bibtex
@misc{grosman2022huggingsound,
  title={{HuggingSound: A toolkit for speech-related tasks based on Hugging Face's tools}},
  author={Grosman, Jonatas},
  howpublished={\url{https://github.com/jonatasgrosman/huggingsound}},
  year={2022}
}
```
