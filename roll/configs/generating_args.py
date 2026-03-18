from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, List

@dataclass
class GeneratingArguments:
    r"""
    Arguments pertaining to specify the decoding parameters.
    """

    do_sample: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether or not to use sampling, use greedy decoding otherwise."},
    )
    temperature: Optional[float] = field(
        default=0.95,
        metadata={"help": "The value used to modulate the next token probabilities."},
    )
    top_p: Optional[float] = field(
        default=0.7,
        metadata={
            "help": "The smallest set of most probable tokens with probabilities that add up to top_p or higher are kept."
        },
    )
    top_k: Optional[int] = field(
        default=50,
        metadata={"help": "The number of highest probability vocabulary tokens to keep for top-k filtering."},
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={"help": "Number of beams for beam search. 1 means no beam search."},
    )
    max_length: Optional[int] = field(
        default=8192,
        metadata={"help": "The maximum length the generated tokens can have. It can be overridden by max_new_tokens."},
    )
    max_new_tokens: Optional[int] = field(
        default=8192,
        metadata={"help": "The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt."},
    )
    repetition_penalty: Optional[float] = field(
        default=1.0,
        metadata={"help": "The parameter for repetition penalty. 1.0 means no penalty."},
    )
    length_penalty: Optional[float] = field(
        default=1.0,
        metadata={"help": "Exponential penalty to the length that is used with beam-based generation."},
    )
    num_return_sequences: Optional[int] = field(
        default=1,
        metadata={"help": "The number of independently computed returned sequences for each element in the batch."},
    )
    stop_words: Optional[List[str]] = field(    
        default_factory=lambda: [""],
        metadata={"help": "The words to stop generation."},
    )
    stop_strings: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": (
                "Alias of stop_words. Kept for backward-compatibility with existing YAML configs "
                "that use `stop_strings`."
            )
        },
    )

    def to_dict(self) -> Dict[str, Any]:
        args = asdict(self)

        stop_strings = args.get("stop_strings") or []
        stop_words = args.get("stop_words") or []
        if stop_strings:
            if stop_words == [""] or not any(w and w.strip() for w in stop_words):
                args["stop_words"] = list(stop_strings)
            else:
                merged = list(stop_words)
                for s in stop_strings:
                    if s not in merged:
                        merged.append(s)
                args["stop_words"] = merged
        args.pop("stop_strings", None)

        if args.get("max_new_tokens", -1) > 0:
            args.pop("max_length", None)
        else:
            args.pop("max_new_tokens", None)
        return args
