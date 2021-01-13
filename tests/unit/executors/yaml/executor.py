from typing import Optional

import numpy as np

from jina.executors.decorators import batching, as_ndarray
from jina.executors.devices import TorchDevice
from jina.executors.encoders import BaseEncoder


class DummyTransformerTorchEncoder(TorchDevice, BaseEncoder):
    """
    Internally, TransformerTorchEncoder wraps the pytorch version of transformers from huggingface.
    """

    def __init__(
            self,
            pretrained_model_name_or_path: str = 'sentence-transformers/distilbert-base-nli-stsb-mean-tokens',
            base_tokenizer_model: Optional[str] = None,
            pooling_strategy: str = 'mean',
            layer_index: int = -1,
            max_length: Optional[int] = None,
            acceleration: Optional[str] = None,
            model_save_path: Optional[str] = None,
            *args,
            **kwargs,
    ):

        super().__init__(*args, **kwargs)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.base_tokenizer_model = base_tokenizer_model or pretrained_model_name_or_path
        self.pooling_strategy = pooling_strategy
        self.layer_index = layer_index
        self.max_length = max_length
        self.acceleration = acceleration
        self.model_save_path = model_save_path

        if self.pooling_strategy == 'auto':
            self.pooling_strategy = 'cls'
            self.logger.warning(
                '"auto" pooling_strategy is deprecated, Defaulting to '
                ' "cls" to maintain the old default behavior.'
            )

        if self.pooling_strategy not in ['cls', 'mean', 'max', 'min']:
            self.logger.error(
                f'pooling strategy not found: {self.pooling_strategy}.'
                ' The allowed pooling strategies are "cls", "mean", "max", "min".'
            )
            raise NotImplementedError

        if self.acceleration not in [None, 'amp', 'quant']:
            self.logger.error(
                f'acceleration not found: {self.acceleration}.'
                ' The allowed accelerations are "amp" and "quant".'
            )
            raise NotImplementedError

    @property
    def model_abspath(self) -> str:
        """Get the file path of the encoder model storage"""
        return self.get_file_from_workspace(self.model_save_path)

    def post_init(self):
        pass

    @batching
    @as_ndarray
    def encode(self, data: 'np.ndarray', *args, **kwargs) -> 'np.ndarray':
        """
        :param data: a 1d array of string type in size `B`
        :return: an ndarray in size `B x D`
        """
        pass
