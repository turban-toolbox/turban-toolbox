import pytest
from pydantic import ValidationError

from turban.process.utemp.config import FP07_RESPONSE_MODELS, UTempConfig

SEGMENT_KWARGS = dict(
    sampfreq=1024.0,
    segment_length=512,
    segment_overlap=256,
    chunk_length=2048,
    chunk_overlap=1024,
)


@pytest.mark.parametrize("name", sorted(FP07_RESPONSE_MODELS))
def test_resolve_fp07_response_from_name(name):
    cfg = UTempConfig(**SEGMENT_KWARGS, fp07_response=name)
    assert cfg.fp07_response == FP07_RESPONSE_MODELS[name]


def test_resolve_fp07_response_unknown_name():
    with pytest.raises(ValidationError, match="Unknown FP07 response model"):
        UTempConfig(**SEGMENT_KWARGS, fp07_response="not_a_model")
