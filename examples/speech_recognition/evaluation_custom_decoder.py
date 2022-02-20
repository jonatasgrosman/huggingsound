import torch
from huggingsound import SpeechRecognitionModel, KenshoLMDecoder

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 1
model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english", device=device)

references = [
    {"path": "/path/to/sagan.mp3", "transcription": "extraordinary claims require extraordinary evidence"},
    {"path": "/path/to/asimov.wav", "transcription": "violence is the last refuge of the incompetent"},
]

# The LM format used by the LM decoders is the KenLM format (arpa or binary file).
# You can download some LM files examples from here: https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english/tree/main/language_model
lm_path = "path/to/your/lm_files/lm.binary"
unigrams_path = "path/to/your/lm_files/unigrams.txt"

# To use this decoder you'll need to install the Kensho's ctcdecode first (https://github.com/kensho-technologies/pyctcdecode)
decoder = KenshoLMDecoder(model.token_set, lm_path=lm_path, unigrams_path=unigrams_path, alpha=2, beta=1, beam_width=100)

evaluation = model.evaluate(references, inference_batch_size=batch_size, decoder=decoder)

print(evaluation)