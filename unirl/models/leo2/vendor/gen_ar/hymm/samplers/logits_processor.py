# mostly copied from transformers repo, we define logits processor in here for further Implementation
# https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py

import inspect
import torch


class LogitsProcessor:
    """Abstract base class for all logit processors that can be applied during generation."""
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    def maybe_update(self, **kwargs):
        pass


class TemperatureLogitsWarper(LogitsProcessor):
    r"""
    [`LogitsProcessor`] for temperature (exponential scaling output probability distribution), which effectively means
    that it can control the randomness of the predicted tokens. Often used together with [`TopPLogitsWarper`] and
    [`TopKLogitsWarper`].

    <Tip>

    Make sure that `do_sample=True` is included in the `generate` arguments otherwise the temperature value won't have
    any effect.

    </Tip>

    Args:
        temperature (`float`):
            Strictly positive float value used to modulate the logits distribution. A value smaller than `1` decreases
            randomness (and vice versa), with `0` being equivalent to shifting all probability mass to the most likely
            token.

    Examples:

    ```python
    >>> import torch
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> set_seed(0)  # for reproducibility

    >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    >>> model.config.pad_token_id = model.config.eos_token_id
    >>> inputs = tokenizer(["Hugging Face Company is"], return_tensors="pt")

    >>> # With temperature=1.0, the default, we consistently get random outputs due to random sampling.
    >>> generate_kwargs = {"max_new_tokens": 10, "do_sample": True, "temperature": 1.0, "num_return_sequences": 2}
    >>> outputs = model.generate(**inputs, **generate_kwargs)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    ['Hugging Face Company is one of these companies that is going to take a',
    "Hugging Face Company is a brand created by Brian A. O'Neil"]

    >>> # However, with temperature close to 0, it approximates greedy decoding strategies (invariant)
    >>> generate_kwargs["temperature"] = 0.0001
    >>> outputs = model.generate(**inputs, **generate_kwargs)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    ['Hugging Face Company is a company that has been around for over 20 years',
    'Hugging Face Company is a company that has been around for over 20 years']
    ```
    """

    def __init__(self, temperature: float):
        self.check_args(temperature)
        self.temperature = temperature

    @staticmethod
    def check_args(temperature: float):
        if not isinstance(temperature, float) or not (temperature > 0):
            except_msg = (
                f"`temperature` (={temperature}) has to be a strictly positive float, otherwise your next token "
                "scores will be invalid."
            )
            if isinstance(temperature, float) and temperature == 0.0:
                except_msg += " If you're looking for greedy decoding strategies, set `do_sample=False`."
            raise ValueError(except_msg)

    def maybe_update(self, **kwargs):
        if 'temperature' in kwargs:
            self.check_args(kwargs['temperature'])
            self.temperature = kwargs['temperature']

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores_processed = scores / self.temperature
        return scores_processed

    def __repr__(self):
        return f"TemperatureLogitsWarper(temperature={self.temperature})"


class TopPLogitsWarper(LogitsProcessor):
    """
    [`LogitsProcessor`] that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <= prob_cut_off.
    Often used together with [`TemperatureLogitsWarper`] and [`TopKLogitsWarper`].

    Args:
        top_p (`float`):
            If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
            higher are kept for generation.
        filter_value (`float`, *optional*, defaults to -inf):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> set_seed(1)
    >>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

    >>> inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")

    >>> # With sampling, the output is unexpected -- sometimes too unexpected.
    >>> outputs = model.generate(**inputs, do_sample=True)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 3 | < 4 (left-hand pointer) ;
    <BLANKLINE>
    <BLANKLINE>

    >>> # With `top_p` sampling, the output gets restricted to high-probability tokens.
    >>> # Pro tip: In practice, LLMs use `top_p` in the 0.9-0.95 range.
    >>> outputs = model.generate(**inputs, do_sample=True, top_p=0.1)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9
    ```
    """

    def __init__(self, top_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def maybe_update(self, **kwargs):
        if 'top_p' in kwargs:
            top_p = float(kwargs['top_p'])
            if top_p < 0 or top_p > 1.0:
                raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")
            self.top_p = top_p

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed

    def __repr__(self):
        return f"TopPLogitsWarper(top_p={self.top_p})"


class TopKLogitsWarper(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that performs top-k, i.e. restricting to the k highest probability elements. Often used
    together with [`TemperatureLogitsWarper`] and [`TopPLogitsWarper`].

    Args:
        top_k (`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (`float`, *optional*, defaults to -inf):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> set_seed(1)
    >>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

    >>> inputs = tokenizer("A sequence: A, B, C, D", return_tensors="pt")

    >>> # With sampling, the output is unexpected -- sometimes too unexpected.
    >>> outputs = model.generate(**inputs, do_sample=True)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: A, B, C, D, E — S — O, P — R

    >>> # With `top_k` sampling, the output gets restricted the k most likely tokens.
    >>> # Pro tip: In practice, LLMs use `top_k` in the 5-50 range.
    >>> outputs = model.generate(**inputs, do_sample=True, top_k=2)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: A, B, C, D, E, F, G, H, I
    ```
    """

    def __init__(self, top_k: int, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")

        self.min_tokens_to_keep = min_tokens_to_keep
        self.top_k = max(top_k, min_tokens_to_keep)
        self.filter_value = filter_value

    def maybe_update(self, **kwargs):
        if 'top_k' in kwargs:
            self.top_k = max(kwargs['top_k'], self.min_tokens_to_keep)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        top_k = min(self.top_k, scores.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed

    def __repr__(self):
        return f"TopKLogitsWarper(top_k={self.top_k})"


class SliceVocabLogitsWarper(LogitsProcessor):
    """
    [`LogitsProcessor`] that performs vocab slicing, i.e. restricting probabilities with in some range. This processor
    is often used in multimodal discrete LLMs, which ensure that we only sample within one modality

    Args:
        vocab_start (`int`): start of slice, default None meaning from 0
        vocab_end (`int`): end of slice, default None meaning to the end of list
        when start and end are all None, this processor does noting

    """

    def __init__(self, vocab_start: int = None, vocab_end: int = None, **kwargs):
        if vocab_start is not None and vocab_end is not None:
            assert vocab_start < vocab_end, f"Ensure vocab_start {vocab_start} < vocab_end {vocab_end}"
        self.vocab_start = vocab_start
        self.vocab_end = vocab_end
        self.other_slices = kwargs.get("other_slices", [])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores_processed = scores[:, self.vocab_start: self.vocab_end]
        for other_slice in self.other_slices:
            scores_processed = torch.cat([scores_processed, scores[:, other_slice[0]: other_slice[1]]], dim=-1)
        return scores_processed

    def __repr__(self):
        return f"SliceVocabLogitsWarper(vocab_start={self.vocab_start}, vocab_end={self.vocab_end}, other_slices={self.other_slices})"


class MaskVocabLogitsWarper(LogitsProcessor):
    """Restrict generation to a vocab slice while preserving original token ids."""

    def __init__(self, vocab_start: int = None, vocab_end: int = None, **kwargs):
        if vocab_start is not None and vocab_end is not None:
            assert vocab_start < vocab_end, f"Ensure vocab_start {vocab_start} < vocab_end {vocab_end}"
        self.vocab_start = vocab_start
        self.vocab_end = vocab_end
        self.other_slices = kwargs.get("other_slices", [])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores_processed = torch.full_like(scores, -float("Inf"))
        scores_processed[:, self.vocab_start: self.vocab_end] = scores[:, self.vocab_start: self.vocab_end]
        for other_slice in self.other_slices:
            scores_processed[:, other_slice[0]: other_slice[1]] = scores[:, other_slice[0]: other_slice[1]]
        return scores_processed

    def __repr__(self):
        return f"MaskVocabLogitsWarper(vocab_start={self.vocab_start}, vocab_end={self.vocab_end}, other_slices={self.other_slices})"


class SliceDoubleVocabLogitsWarper(LogitsProcessor):
    """
    [`LogitsProcessor`] that performs vocab slicing, i.e. restricting probabilities with in some range. This processor
    is often used in multimodal discrete LLMs, which ensure that we only sample within one modality

    Args:
        vocab_start (`int`): start of slice, default None meaning from 0
        vocab_end (`int`): end of slice, default None meaning to the end of list
        when start and end are all None, this processor does noting

    """

    def __init__(self, num_vocab, vocab_start: list, vocab_end: list):
        if vocab_start is not None and vocab_end is not None:
            assert vocab_start < vocab_end, f"Ensure vocab_start {vocab_start} < vocab_end {vocab_end}"
        self.vocab_start = vocab_start
        self.vocab_end = vocab_end
        self.num_vocab = num_vocab
        assert len(vocab_start) == len(vocab_end) == num_vocab, f"Ensure length of vocab start and end equals to num_vocab"
        self.count = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        idx = self.count % self.num_vocab
        scores_processed = scores[:, self.vocab_start[idx]: self.vocab_end[idx]]
        self.count += 1
        if self.count == self.num_vocab:
            self.count = 0      # reset to zero when interleaved sequence hit ends
        return scores_processed
    
    def __repr__(self):
        return f"SliceDoubleVocabLogitsWarper(vocab_start={self.vocab_start}, vocab_end={self.vocab_end})"


class CfgLogitsWarper(LogitsProcessor):
    def __init__(self, guidance_scale: float):
        self.guidance_scale = guidance_scale
    
    def __call__(self, input_ids: torch.LongTensor, logits: torch.FloatTensor):
        # input_ids.shape=[2*batch, seq-len]
        # logits.shape=[2*batch, vocab]
        conditioned_logits, unconditioned_logits = torch.chunk(logits, chunks=2, dim=0)
        mixed_logits = unconditioned_logits + self.guidance_scale * (
            conditioned_logits - unconditioned_logits
        )
        return mixed_logits.repeat(2, 1)
    
    def __repr__(self) -> str:
        return f"CfgLogitsWarper(guidance_scale={self.guidance_scale})"


class LogitsProcessorList(list):
    """
    This class can be used to create a list of [`LogitsProcessor`] to subsequently process a `scores` input tensor.
    This class inherits from list and adds a specific *__call__* method to apply each [`LogitsProcessor`] to the
    inputs.
    """

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            kwargs (`Dict[str, Any]`, *optional*):
                Additional kwargs that are specific to a logits processor.

        Return:
            `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`:
                The processed prediction scores.

        """
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 2:
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} for "
                        f"{processor.__class__} are passed to the logits processor."
                    )
                scores = processor(input_ids, scores, **kwargs)
            else:
                scores = processor(input_ids, scores)

        return scores

    def __repr__(self):
        output = "LogitsProcessorList(\n"
        for processor in self:
            output += ' ' * 30 + str(processor) + "\n"
        output += ' ' * 26 + ")"
        return output

    def update(self, **kwargs):
        for processor in self:
            processor.maybe_update(**kwargs)

    def get_processor(self, name):
        name_mapper = {
            processor.__class__.__name__: processor
            for processor in self
        }
        return name_mapper.get(name)


def AutoLogitsProcessorFromCfg(cfg: dict):
    assert len(cfg) == 1, f"cfg should only have one key, got {len(cfg)}"
    for name, kwargs in cfg.items():
        logits_wrapper = globals()[name](**kwargs)
    return logits_wrapper


def get_logits_processors(cfg: list):
    logits_processor_list = LogitsProcessorList()
    for logits_cfg in cfg:
        logits_processor_list.append(AutoLogitsProcessorFromCfg(logits_cfg))
    return logits_processor_list


def update_logits_processor_kwargs(logits_processor_list, top_k=None, top_p=None, guidance_scale=None):
    for processor in logits_processor_list:
        if isinstance(processor, TopKLogitsWarper):
            if top_k is not None:
                processor.top_k = top_k
        elif isinstance(processor, TopPLogitsWarper):
            if top_p is not None:
                processor.top_p = top_p
        elif isinstance(processor, CfgLogitsWarper):
            if guidance_scale is not None:
                processor.guidance_scale = guidance_scale
    return logits_processor_list
