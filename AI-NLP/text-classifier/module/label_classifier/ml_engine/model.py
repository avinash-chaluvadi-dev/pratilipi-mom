import torch.nn as nn

from .. import config


class LabelBackbone(nn.Module):
    """
    LabelBackbone - Backbone model class for Label Classification.
    """

    def __init__(self, model, tokenizer):
        super(LabelBackbone, self).__init__()
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
