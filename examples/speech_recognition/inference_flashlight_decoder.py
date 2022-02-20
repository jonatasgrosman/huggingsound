import torch
from huggingsound import SpeechRecognitionModel, FlashlightLMDecoder

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 1
model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english", device=device)
audio_paths = ["/path/to/sagan.mp3", "/path/to/asimov.wav"]

# The LM format used by the LM decoders is the KenLM format (arpa or binary file).
# You can download some LM files examples from here: https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english/tree/main/language_model
lm_path = "path/to/your/lm_files/lm.binary"
unigrams_path = "path/to/your/lm_files/unigrams.txt"
lexicon_path = "path/to/your/lm_files/lexicon.txt"

# you can generate a lexicon file from a unigram file using the following command:
with open(lexicon_path, "w") as f_lexicon:
    with open(unigrams_path, "r") as f_unigrams:
        for w in f_unigrams:
            f_lexicon.write(w.strip() + "\t" + "".join(list(map(lambda a: a + " ", list(w.strip())))) + "|\n")

# To use this decoder you'll need to install the Flashlight's ctcdecode first (https://github.com/flashlight/flashlight/tree/main/bindings/python)
decoder = FlashlightLMDecoder(model.token_set, lm_path=lm_path, lexicon_path=lexicon_path, lm_weight=2, word_score=1, beam_size=100)

transcriptions = model.transcribe(audio_paths, batch_size=batch_size, decoder=decoder)

print(transcriptions)
