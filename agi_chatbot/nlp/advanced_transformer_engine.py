"""Stub for nlp.advanced_transformer_engine used by api_server.

Provide `get_transformer_engine` and `TransformerConfig` used by
`api_server` initialization code.
"""


class TransformerConfig:
    def __init__(self, model_name: str = None, use_cuda: bool = False, **kwargs):
        self.model_name = model_name
        self.use_cuda = bool(use_cuda)


def get_transformer_engine(config: TransformerConfig = None, *args, **kwargs):
    class Engine:
        def __init__(self, cfg):
            self.cfg = cfg

        def infer(self, text: str):
            return text

    return Engine(config)


def transform_text(text: str):
    return text
