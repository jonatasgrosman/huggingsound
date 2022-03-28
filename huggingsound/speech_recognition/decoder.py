from __future__ import annotations
from typing import Optional, Union
import torch
import logging
from multiprocessing import Pool
import numpy as np
from huggingsound.token_set import TokenSet

class Decoder():
    """
    Decoder

    Parameters
    ----------
    token_set : TokenSet
        The TokenSet object to use for decoding.

    skip_special_tokens: Optional[bool] = True
        If True, skip the special tokens in the TokenSet during decoding.
    
    ms_per_timestep : Optional[int] = 20
        The number of milliseconds per timestep.

        The magic number 20 comes from the Wav2Vec2 convolutional layers. I'll try to explain this a little:
            The convolutional layers of the feature extractor have padding=0, kernel=(10,3,3,3,3,2,2) and strides=(5,2,2,2,2,2,2).
            So to map the model output (in timesteps) to the original audio (in milliseconds) we need to apply 
            the convolution output size formula ( ((input-kernel+2*padding)/stride)+1) ) on each layer, given some waveform input.
            However, I don't recommend that approach because it needs some extra computation and it's not really precise, 
            because there're a lot of overlappings due to the kernel and stride sizes desynchronization. 
            So a good and simple approximation can be extracted using just the convolutional layers strides, more specifically, 
            using the product of the strides:

            MS_PER_TIMESTEP = STRIDES_PRODUCT / WAVEFORM_POINTS_PER_MS
            MS_PER_TIMESTEP = 5*2^6 / 16 <- remember that the wav2vec input is in 16000Hz
            MS_PER_TIMESTEP = 20 <- the magic number :)

    probability_offset : Optional[float] = 1
        The probability offset to use when calculating the probability of a token when the end_timestep of a token isn't provided, 
        i. e., how many timesteps to look around to calc the probability.
    
    """

    def __init__(self, token_set: TokenSet, skip_special_tokens: Optional[bool] = True, ms_per_timestep: Optional[int] = 20, 
                 probability_offset: Optional[float] = 1):

        self.token_set = token_set
        self.skip_special_tokens = skip_special_tokens
        self.ms_per_timestep = ms_per_timestep
        self.probability_offset = probability_offset

    def _get_predictions(self, logits: torch.Tensor) -> list[dict]:
        """ 
        Get the predicted ids given the model's output logits.

        Parameters:
        ----------
            logits: torch.Tensor
                Model's output tensors of shape (BATCH_SIZE, TIMESTEPS, TOKEN_SET_SIZE)
        
        Returns:
        ----------
            list[dict]: prediction list of size BATCH_SIZE with format:
                [{
                    "ids": list,
                    "start_timesteps": list,
                    "end_timesteps": list,
                }, ...]
        """

        raise NotImplementedError()

    def _ctc_decode(self, predicted_ids: Union[torch.Tensor, np.ndarray, list[int]], return_timesteps: Optional[bool] = True) -> list[dict]:
        """ 
        Decode the predicted using the CTC decoding algorithm.

        Parameters:
        ----------
            predicted_ids: Union[torch.Tensor, np.ndarray, list[int]]
                Model's output tensors of shape (BATCH_SIZE, TIMESTEPS, TOKEN_SET_SIZE)
            
            return_timesteps: Optional[bool] = True
                If True, return the timesteps of the decoded sequence. 
        
        Returns:
        ----------
            list[dict]: decoded prediction list of size BATCH_SIZE with format:
                [{
                    "ids": list,
                    "start_timesteps": list,
                    "end_timesteps": list,
                }, ...]
        """

        predictions = []

        for i in range(len(predicted_ids)): # for each item in the batch
            
            i_predicted_ids = []
            i_start_timesteps = []
            i_end_timesteps = []
            previous_predicted_id = None
            
            for t in range(len(predicted_ids[i])): # for each timestep in the item

                predicted_id = int(predicted_ids[i][t])
                
                if predicted_id != self.token_set.blank_token_id:
                    
                    if len(i_predicted_ids) == 0 or previous_predicted_id == self.token_set.blank_token_id or predicted_id != i_predicted_ids[-1]:
                        i_predicted_ids.append(predicted_id)
                        if return_timesteps:
                            i_start_timesteps.append(t)
                            i_end_timesteps.append(t+1)

                    elif predicted_id == i_predicted_ids[-1] and return_timesteps:
                        i_end_timesteps[-1] = t

                previous_predicted_id = predicted_id
            
            predictions.append({
                "ids": i_predicted_ids,
                "start_timesteps": i_start_timesteps if return_timesteps else None,
                "end_timesteps": i_end_timesteps if return_timesteps else None,
            })

        return predictions

    def __call__(self, logits: torch.Tensor) -> list[dict]:
        """ 
        Getting the predictions given the model's output logits.

        Parameters:
        ----------
            logits: torch.Tensor
                Model's output tensors of shape (BATCH_SIZE, TIMESTEPS, TOKEN_SET_SIZE)
        
        Returns:
        ----------
            list[dict]: Decoded prediction list of size BATCH_SIZE with format:
                [{
                    "transcription": list,
                    "start_timesteps": list,
                    "end_timesteps": list,
                }, ...]
        """

        result = []
        
        predictions = self._get_predictions(logits)
        logits_probs = torch.nn.functional.softmax(logits.float(), dim=-1).to("cpu").detach()

        for i, prediction in enumerate(predictions):

            transcription = ""
            transcription_start_timestamps = []
            transcription_end_timestamps = []
            transcription_probabilities = []

            if "transcription" in prediction:
                transcription = prediction["transcription"]
                J = range(len(transcription))
            else:
                J = range(len(prediction["ids"]))

            for j in J:

                if "transcription" in prediction:
                    token = transcription[j] if transcription[j] != " " else self.token_set.silence_token
                    if token not in self.token_set.tokens:
                        token = self.token_set.unk_token
                    predicted_id = self.token_set.id_by_token[token]
                else:
                    predicted_id = prediction["ids"][j]

                    if predicted_id == self.token_set.silence_token_id:
                        transcription += " "
                    elif self.skip_special_tokens and self.token_set.tokens[predicted_id] in self.token_set.special_tokens:
                        continue
                    else:
                        transcription += self.token_set.tokens[predicted_id]
                
                if prediction["start_timesteps"] is not None:
                    start_timestep = prediction["start_timesteps"][j]
                    transcription_start_timestamps.append(int(start_timestep * self.ms_per_timestep))

                    # as we report the character based probability and more than one timestep can be responsable for the character prediction,
                    # when a start_timestep and end_timestep are provided we'll report the mean value of the this range of timesteps,
                    # otherwise we'll report the mean probability of a window defined be the start_timestep_t and start_timestep_t+1

                    if prediction["end_timesteps"] is not None:
                        window_end_timestep = prediction["end_timesteps"][j]
                    else:
                        window_end_timestep = start_timestep + 1
                    
                    if start_timestep == window_end_timestep: # it needs to have at least one timestep of difference
                        window_end_timestep += 1

                    window_probabilities = [x[predicted_id] for x in logits_probs[i][start_timestep:window_end_timestep]]
                    probability = float(np.mean(window_probabilities))
                    transcription_probabilities.append(probability)

                if prediction["end_timesteps"] is not None:
                    end_timestep = prediction["end_timesteps"][j]
                    transcription_end_timestamps.append(int(end_timestep * self.ms_per_timestep))

                    #probability = float(logits_probs[i][start_timestep][predicted_id])
                    #transcription_probabilities[-1] = probability

            # transcription trimming
            if len(transcription) > 0:
                
                left_offset = len(transcription) - len(transcription.lstrip())
                right_offset = len(transcription) - len(transcription.rstrip())

                transcription = transcription[left_offset:len(transcription)-right_offset]

                if len(transcription_start_timestamps) > 0:
                    transcription_start_timestamps = transcription_start_timestamps[left_offset:len(transcription_start_timestamps)-right_offset]
                if len(transcription_end_timestamps) > 0:
                    transcription_end_timestamps = transcription_end_timestamps[left_offset:len(transcription_end_timestamps)-right_offset]
                if len(transcription_probabilities) > 0:    
                    transcription_probabilities = transcription_probabilities[left_offset:len(transcription_probabilities)-right_offset]
            
            result.append({
                "transcription": transcription,
                "start_timestamps": transcription_start_timestamps if len(transcription_start_timestamps) > 0 else None,
                "end_timestamps": transcription_end_timestamps if len(transcription_end_timestamps) > 0 else None,
                "probabilities": transcription_probabilities if len(transcription_probabilities) > 0 else None,
            })
        
        return result


class GreedyDecoder(Decoder):
    """
    Greedy decoder

    Parameters
    ----------
    token_set : TokenSet
        The TokenSet object to use for decoding.
    """

    def __init__(self, token_set: TokenSet):
        super().__init__(token_set)

    def _get_predictions(self, logits: torch.Tensor):

        predicted_ids = torch.argmax(logits, dim=-1)
        predictions = self._ctc_decode(predicted_ids)

        return predictions


class ParlanceLMDecoder(Decoder):
    """
    Parlance Language Model decoder

    Parameters
    ----------
    token_set : TokenSet
        The TokenSet object to use for decoding.
    
    lm_path : str
        Path to the KenLM language model file
    
    alpha: Optional[float] = 2.0
        Weighting associated with the LMs probabilities. A weight of 0 means the LM has no effect.
    
    beta: Optional[float] = -1.0
        Weight associated with the number of words within our beam (LM usage reward).
    
    cutoff_top_n: Optional[int] = 40
        Cutoff number in pruning. Only the top cutoff_top_n characters with the highest probability in 
        the TokenSet will be used in beam search.
    
    cutoff_prob: Optional[float] = 1.0
        Cutoff probability in pruning. 1.0 means no pruning.
    
    beam_width: Optional[int] = 100
        This controls how broad the beam search is. Higher values are more likely to find top beams, 
        but they also will make your beam search exponentially slower. Furthermore, the longer your outputs, 
        the more time large beams will take. This is an important parameter that represents a tradeoff you need 
        to make based on your dataset and needs.
    
    num_processes: Optional[int] = 4
        Parallelize the batch using num_processes workers. You probably want to pass the number of cpus your computer has. 
        You can find this in python with import multiprocessing then n_cpus = multiprocessing.cpu_count()
    """

    def __init__(self, token_set: TokenSet, lm_path: str, alpha: Optional[float] = 2.0, beta: Optional[float] = -1.0, 
                 cutoff_top_n: Optional[int] = 40, cutoff_prob: Optional[float] = 1.0, beam_width: Optional[int] = 100, 
                 num_processes: Optional[int] = 4):
        
        super().__init__(token_set)

        self.lm_path = lm_path
        self.alpha = alpha
        self.beta = beta
        self.cutoff_top_n = cutoff_top_n
        self.cutoff_prob = cutoff_prob
        self.beam_width = beam_width
        self.num_processes = num_processes

        try:
            from ctcdecode import CTCBeamDecoder
        except ImportError:
            raise ImportError("To use this decoder please install the ctcdecoder from https://github.com/parlance/ctcdecode")

        # creating the tokens forcing the silence token to be a whitespace
        tokens = [x if x != self.token_set.silence_token else " " for x in self.token_set.tokens]
        
        self.ctcdecoder = CTCBeamDecoder(tokens, 
            model_path=self.lm_path,
            alpha=self.alpha,
            beta=self.beta,
            cutoff_top_n=self.cutoff_top_n,
            cutoff_prob=self.cutoff_prob,
            beam_width=self.beam_width,
            num_processes=self.num_processes,
            blank_id=self.token_set.blank_token_id,
            
            # If your outputs have passed through a softmax and represent probabilities, this should be false, 
            # if they passed through a LogSoftmax and represent negative log likelihood, you need to pass True. 
            # If you don't understand this, run print(output[0][0].sum()), if it's a negative number you've probably got NLL 
            # and need to pass True, if it sums to ~1.0 you should pass False. We set this to True by default, 
            # 'cause nowadays the logits are not normalized and we'll apply a softmax to them before decoding.
            log_probs_input=True
        )

    def _get_predictions(self, logits: torch.Tensor) -> list[dict]:

        # beam_results (BATCHSIZE x N_BEAMS X N_TIMESTEPS): A batch containing the series 
        # of characters (these are ints, you still need to decode them back to your text) representing 
        # results from a given beam search. Note that the beams are almost always shorter than the 
        # total number of timesteps, and the additional data is non-sensical, so to see the top beam 
        # (as int labels) from the first item in the batch, you need to run beam_results[0][0][:out_len[0][0]].

        # beam_scores (BATCHSIZE x N_BEAMS): A batch with the approximate CTC score of each beam 
        # If this is true, you can get the model's confidence that the beam is correct with p=1/np.exp(beam_score).

        # beam_timesteps (BATCHSIZE x N_BEAMS X N_TIMESTEPS) the timestep at which the nth output character has peak probability. 
        # Can be used as alignment between the audio and the transcript.

        # beam_end_lens (BATCHSIZE x N_BEAMS): out_lens[i][j] is the length of the jth beam_result, of item i of your batch.
        
        beam_results, beam_scores, beam_timesteps, beam_end_lens = self.ctcdecoder.decode(logits.to("cpu"))

        predictions = []

        B, N, T = beam_results.size()

        for i in range(B):
            length = beam_end_lens[i][0]

            predictions.append({
                "ids": beam_results[i][0][:length],
                "start_timesteps": beam_timesteps[i][0][:length],
                "end_timesteps": None,
            })

        return predictions


class FlashlightLMDecoder(Decoder):
    """
    Flashlight Language Model decoder

    Parameters
    ----------
    token_set : TokenSet
        The TokenSet object to use for decoding.
    
    lm_path : str
        Path to the KenLM language model file
    
    lexicon_path: Optional[str] = None
        Path to the lexicon file
    
    beam_size: Optional[int] = 100
        Number of top hypothesis to preserve at each decoding step.
    
    beam_threshold: Optional[float] = 100.0
        Preserve a hypothesis only if its score is not far away from the current best hypothesis score.
    
    beam_size_token: Optional[int] = None
        Restrict number of tokens by top am scores (if you have a huge token set). If None, the beam size is len(TokenSet)
    
    lm_weight: Optional[float] = 2.0
        Language model weight for LM score.
    
    word_score: Optional[float] = -1.0
        Score for words appearance in the transcription (word insertion penalty).
    
    sil_score: Optional[float] = 0.0
        Score for silence appearance in the transcription.
    
    decoder_device: Optional[str] = "cpu"
        The device where the decoder will be executed, by default it's "cpu", if you want to use gpu, 
        then you'll need to install the flashlight with CUDA support and set the decoder_device to "cuda"
    
    """

    def __init__(self, token_set: TokenSet, lm_path: str, lexicon_path: Optional[str] = None, beam_size: Optional[int] = 100, 
                 beam_threshold: Optional[float] = 100.0, beam_size_token: Optional[int] = None, lm_weight: Optional[float] = 2.0,
                 word_score: Optional[float] = -1.0, sil_score: Optional[float] = 0.0, decoder_device: Optional[str] = "cpu"):
        
        super().__init__(token_set)
                
        try:
            from flashlight.lib.text.dictionary import create_word_dict, load_words
            from flashlight.lib.text.decoder import (
                CriterionType,
                LexiconDecoderOptions,
                KenLM,
                SmearingMode,
                Trie,
                LexiconDecoder,
            )
            from flashlight.lib.text.decoder import LexiconFreeDecoder, LexiconFreeDecoderOptions

        except:
            raise ImportError("To use this decoder please install the flashlight python bindings from https://github.com/facebookresearch/flashlight/tree/master/bindings/python")
        
        self.lm_path = lm_path
        self.lexicon_path = lexicon_path
        self.beam_size = beam_size
        self.beam_threshold = beam_threshold
        self.beam_size_token = beam_size_token if beam_size_token is not None else self.token_set.size
        self.lm_weight = lm_weight
        self.word_score = word_score
        self.sil_score = sil_score
        self.decoder_device = decoder_device

        if self.lexicon_path is not None:
            tokens_keys = self.token_set.id_by_token.keys()

            # load lexicon file, which defines spelling of words
            # the format word and its tokens spelling separated by the spaces, e.g.:
            # ann a n 1 |
            self.lexicon = load_words(self.lexicon_path)
            
            # read lexicon and store it in the w2l dictionary
            self.word_dict = create_word_dict(self.lexicon)

            # create Kenlm language model
            self.lm = KenLM(self.lm_path, self.word_dict)
            
            # build trie (specifying how many tokens we have and silence index)
            # Trie is necessary to do beam-search decoding with word-level lm
            # We restrict our search only to the words from the lexicon
            # Trie is constructed from the lexicon, each node is a token
            # path from the root to a leaf corresponds to a word spelling in the lexicon
            self.trie = Trie(self.token_set.size, self.token_set.silence_token_id)
            start_state = self.lm.start(False)

            # use heuristic for the trie, called smearing:
            # predict lm score for each word in the lexicon, set this score to a leaf
            # (we predict lm score for each word as each word starts a sentence)
            # word score of a leaf is propagated up to the root to have some proxy score
            # for any intermediate path in the trie
            # SmearingMode defines the function how to process scores
            # in a node came from the children nodes:
            # could be max operation or logadd or none
            for i, (word, spellings) in enumerate(self.lexicon.items()):
                word_idx = self.word_dict.get_index(word)
                _, score = self.lm.score(start_state, word_idx)

                for spelling in spellings:
                    spelling_idxs = []
                    for token in spelling:
                        if token.upper() in tokens_keys:
                            spelling_idxs.append(self.token_set.id_by_token[token.upper()])
                        elif token.lower() in tokens_keys:
                            spelling_idxs.append(self.token_set.id_by_token[token.lower()])
                        else:
                            logging.warning(f"WARNING: The token {token} not exist in your TokenSet, using <unk> token instead")
                            spelling_idxs.append(self.token_set.unk_token_id)
                        
                    self.trie.insert(spelling_idxs, word_idx, score)
            self.trie.smear(SmearingMode.MAX)

            # Define decoder options
            self.decoder_opts = LexiconDecoderOptions(
                beam_size=self.beam_size,
                beam_size_token=self.beam_size_token,
                beam_threshold=self.beam_threshold,
                lm_weight=self.lm_weight,
                word_score=self.word_score,
                unk_score=-np.inf,
                sil_score=self.sil_score,
                log_add=False,
                criterion_type=CriterionType.CTC,
            )

            # define lexicon beam-search decoder with word-level lm
            # LexiconDecoder(decoder options, trie, lm, silence index,
            #                blank index (for CTC), unk index,
            #                transitiona matrix, is token-level lm)
            self.decoder = LexiconDecoder(
                self.decoder_opts,
                self.trie,
                self.lm,
                self.token_set.silence_token_id,
                self.token_set.blank_token_id,
                self.token_set.unk_token_id,
                [],
                False,
            )

        else:
            
            # NOTE: lexicon free decoding can only be properly done with a unit language model
            
            # The general format for each line of the lexicon file is:
            # {representation of the word/unit in the language model}      {representation of ctc model labels used for training}
            
            # So if your model was trained on letters, then the format is:
            # WRITE        W R I T E |
            
            # But if it is phones, it could be:
            # WRITE        WRY  T |

            # And if it's just words, it's basically the identity function:
            # WRITE      WRITE

            # ... so when the representation of the word/unit is the same as the representation of the ctc labels,
            # you have a unit language model, and you can use a lexicon free decoding, 
            # but this decoder usually doesn't give good results, so I don't recommend to use it.

            d = {w: [[w]] for w in self.token_set.id_by_token.keys()}
            self.word_dict = create_word_dict(d)
            self.lm = KenLM(self.lm_path, self.word_dict)
            self.decoder_opts = LexiconFreeDecoderOptions(
                beam_size=self.beam_size,
                beam_size_token=self.beam_size_token,
                beam_threshold=self.beam_threshold,
                lm_weight=self.lm_weight,
                sil_score=self.sil_score,
                log_add=False,
                criterion_type=CriterionType.CTC,
            )
            self.decoder = LexiconFreeDecoder(
                self.decoder_opts, 
                self.lm, 
                self.token_set.silence_token_id, 
                self.token_set.blank_token_id, 
                []
            )

    def _get_predictions(self, logits: torch.Tensor) -> list[dict]:

        # This parameter controls the number of best results to be considered for decoding. 
        # Nowadays, it's 1 by default. In the future, I may change this if I add some randomness for decoding.
        nbest_results = 1
        
        logits = logits.to(self.decoder_device)
        B, T, N = logits.size()

        tokens = []
        # scores = []
        for i in range(B):
            logits_ptr = logits.data_ptr() + 4 * i * logits.stride(0)

            # decoder.decode(emissions, Time, Ntokens)
            # result is a list of sorted hypothesis, 0-index is the best hypothesis
            # each hypothesis is a struct with "score" and "words" representation
            # in the hypothesis and the "tokens" representation
            results = self.decoder.decode(logits_ptr, T, N)

            best_results = results[: nbest_results] # getting the best candidates
            tokens_best = []
            # scores_best = []
            for result in best_results:
                tokens_best.append(result.tokens)
                # scores_best.append(result.score)
            tokens.append(tokens_best)
            # scores.append(scores_best)

        token_array = np.array(tokens, dtype=object).transpose((1, 0, 2))
        # scores_arrray = np.array(scores, dtype=object).transpose()

        predicted_ids = token_array[0][:] # getting best candidates

        predictions = self._ctc_decode(predicted_ids)

        return predictions


class KenshoLMDecoder(Decoder):
    """
    Kensho-technologies' pyctcdecode Language Model decoder

    decoder = build_ctcdecoder(
        labels,
        kenlm_model,
        alpha=0.5,  # tuned on a val set
        beta=1.0,  # tuned on a val set
    )
    text = decoder.decode(logits)

    labels: class containing the labels for input logit matrices
    kenlm_model_path: path to kenlm n-gram language model
    unigrams: list of known word unigrams
    alpha: weight for language model during shallow fusion
    beta: weight for length score adjustment of during scoring
    unk_score_offset: amount of log score offset for unknown tokens
    lm_score_boundary: whether to have kenlm respect boundaries when scoring

    Parameters
    ----------
    token_set : TokenSet
        The TokenSet object to use for decoding.
    
    lm_path : str
        Path to the KenLM language model file
    
    alpha: Optional[float] = 2.0
        Weighting associated with the LMs probabilities. A weight of 0 means the LM has no effect.
    
    beta: Optional[float] = -1.0
        Weight associated with the number of words within our beam (LM usage reward).

    unigrams_path: Optional[str] = None
        The path of the unigrams file.

    unk_score_offset: Optional[float] = -10.0
        Amount of log score offset for unknown tokens.

    lm_score_boundary: Optional[bool] = True
        Whether to have kenlm respect boundaries when scoring.

    beam_width: Optional[int] = 100
        Maximum number of beams at each step in decoding.

    beam_prune_logp: Optional[float] = -10.0
        Beams that are much worse than best beam will be pruned.

    token_min_logp: Optional[float] = -5.0
        Tokens below this logp are skipped unless they are argmax of frame.

    prune_history: Optional[bool] = False
        Whether to filter out beams that are the same over max_ngram history.

    hotwords: Optional[list[str]] = None
        List of words with extra importance, can be OOV for LM.

    hotword_weights: Optional[float]) = 10.0
        Weight factor for hotword importance.

    """

    def __init__(self, token_set: TokenSet, lm_path: str, alpha: Optional[float] = 2.0, beta: Optional[float] = -1.0, 
                 unigrams_path: Optional[str] = None, unk_score_offset: Optional[float] = -10.0, lm_score_boundary: Optional[bool] = True,
                 beam_width: Optional[int] = 100, beam_prune_logp: Optional[float] = -10.0, token_min_logp: Optional[float] = -5.0,
                 prune_history: Optional[bool] = False, hotwords: Optional[list[str]] = None, hotword_weights: Optional[float] = 10.0):
        
        super().__init__(token_set)

        self.lm_path = lm_path
        self.alpha = alpha
        self.beta = beta
        self.unigrams_path = unigrams_path
        self.unk_score_offset = unk_score_offset
        self.lm_score_boundary = lm_score_boundary
        self.beam_width = beam_width
        self.beam_prune_logp = beam_prune_logp
        self.token_min_logp = token_min_logp
        self.prune_history = prune_history
        self.hotwords = hotwords
        self.hotword_weights = hotword_weights

        try:
            from pyctcdecode import build_ctcdecoder
        except ImportError:
            raise ImportError("To use this decoder please install the pyctcdecode from https://github.com/kensho-technologies/pyctcdecode")
        
        self.unigrams = None
        if self.unigrams_path is not None:
            self.unigrams = []
            with open(self.unigrams_path, "r") as f:
                for line in f:
                    self.unigrams.append(line.strip())

        # creating the tokens forcing the silence token to be a whitespace
        tokens = [x if x != self.token_set.silence_token else " " for x in self.token_set.tokens]
        
        self.decoder = build_ctcdecoder(
            tokens,
            self.lm_path,
            alpha=self.alpha,
            beta=self.beta,
            unigrams=self.unigrams,
            unk_score_offset=self.unk_score_offset,
            lm_score_boundary=self.lm_score_boundary
        )

    def _get_predictions(self, logits: torch.Tensor) -> list[dict]:

        with Pool() as pool:

            decoder_outputs = self.decoder.decode_beams_batch(
                pool, logits.cpu().detach().numpy(), beam_width=self.beam_width, beam_prune_logp=self.beam_prune_logp, 
                token_min_logp=self.token_min_logp, prune_history=self.prune_history, 
                hotwords=self.hotwords, hotword_weight=self.hotword_weights
            )

            predictions = []

            for decoder_output in decoder_outputs:

                transcription, word_frames, logit_score, lm_score = decoder_output[0]
                start_timesteps = []
                last_end_timestep = None
                
                for word_frame in word_frames:
                    word = word_frame[0]
                    start_timestep = word_frame[1][0]
                    end_timestep = word_frame[1][1]

                    # as the pyctcdecode doesn't return the character based timestamp, we need to make an approximation of it
                    timestep_per_char = max(1, int((end_timestep - start_timestep)/len(word)))

                    # handling whitespaces
                    if last_end_timestep is not None:
                        start_timesteps.append(last_end_timestep)

                    for i in range(0, len(word)):
                        start_timesteps.append(start_timestep + i*timestep_per_char)

                    last_end_timestep = end_timestep

                predictions.append({
                    "transcription": transcription,
                    "start_timesteps": start_timesteps,
                    "end_timesteps": None,
                })

        return predictions
