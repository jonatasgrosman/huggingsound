import torch
from huggingsound import SpeechRecognitionModel

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 1
model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english", device=device)

references = [
    {"path": "/path/to/sagan.mp3", "transcription": "extraordinary claims require extraordinary evidence"},
    {"path": "/path/to/asimov.wav", "transcription": "violence is the last refuge of the incompetent"},
]

evaluation = model.evaluate(references, inference_batch_size=batch_size)

print(evaluation)