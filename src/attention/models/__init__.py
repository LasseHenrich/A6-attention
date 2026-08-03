"""Model family: RNN, single-layer probe, and transformers.

Re-exports the model classes so callers can write
``from attention.models import Seq2SeqRNN``.  The transformer classes are
added in Tasks G/H; the import is guarded so the package still imports when
``transformer.py`` is only a stub.
"""

from attention.models.base import SeqModel
from attention.models.rnn import Decoder, Encoder, RNNCell, Seq2SeqRNN
from attention.models.wrappers import SingleLayerModel

__all__ = [
    "SeqModel",
    "RNNCell",
    "Encoder",
    "Decoder",
    "Seq2SeqRNN",
    "SingleLayerModel",
]

try:  # transformers arrive in Tasks G/H
    from attention.models.transformer import (  # noqa: F401
        DecoderOnlyTransformer,
        EncoderDecoderTransformer,
        TransformerBlock,
    )

    __all__ += [
        "TransformerBlock",
        "DecoderOnlyTransformer",
        "EncoderDecoderTransformer",
    ]
except ImportError:
    pass
