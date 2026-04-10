from .base import BaseReranker, register_reranker
from contextlib import contextmanager
import logging
import warnings
import numpy as np


@contextmanager
def _dedupe_trust_remote_code_for_tokenizer():
    from sentence_transformers.models import Transformer

    original = Transformer._load_init_kwargs.__func__

    def patched(cls, *args, **kwargs):
        init_kwargs = original(cls, *args, **kwargs)
        tokenizer_args = init_kwargs.get("tokenizer_args")
        if tokenizer_args:
            tokenizer_args = dict(tokenizer_args)
            tokenizer_args.pop("trust_remote_code", None)
            init_kwargs["tokenizer_args"] = tokenizer_args
        return init_kwargs

    setattr(Transformer, "_load_init_kwargs", classmethod(patched))
    try:
        yield
    finally:
        setattr(Transformer, "_load_init_kwargs", classmethod(original))


@register_reranker("local")
class LocalReranker(BaseReranker):
    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer

        # 假设这里有一些日志和警告的设置，根据上下文补充完整
        if self.config.reranker.local.model:
            logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
            warnings.filterwarnings("ignore", category=FutureWarning)

        with _dedupe_trust_remote_code_for_tokenizer():
            encoder = SentenceTransformer(
                self.config.reranker.local.model, trust_remote_code=True
            )
        if self.config.reranker.local.encode_kwargs:
            encode_kwargs = self.config.reranker.local.encode_kwargs
        else:
            encode_kwargs = {}
        s1_feature = encoder.encode(s1,**encode_kwargs,show_progress_bar=True)
        s2_feature = encoder.encode(s2,**encode_kwargs,show_progress_bar=True)
        sim = encoder.similarity(s1_feature, s2_feature)
        return sim.numpy()
