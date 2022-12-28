import torch.nn as nn

from .. import config


class MarkerBackbone(nn.Module):
    """

    MarkerBackbone - Backbone model class for Marker Classifier
        :param model: Transformer model name from config.TRANSFORMER_MODEL_LIST or saved transformer model name
        :param tokenizer: tokenizer from huggingface library
        :returns Object of MarkerBackbone Model

    """

    def __init__(self, model=None, tokenizer=None):
        super(MarkerBackbone, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.drop_out = nn.Dropout(p=0.3)
        self.linear = nn.Linear(768, len(config.CLASSIFICATION_LABELS))

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, po = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False,
        )
        final_output = self.linear(self.drop_out(po))
        return final_output
