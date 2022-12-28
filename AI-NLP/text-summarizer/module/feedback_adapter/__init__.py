from .modules.rectifier import TransRectifier, rectifier_validator


def feedback_cleaner(audited_json, rules_json):

    # Rectifier Components.
    trans_rect = TransRectifier(audited_json["concatenated_view"], rules_json)
    generated_rect = trans_rect.rectify()

    # Label Annotation Components.
    # trans_annot = TransAnnotator(audited_json["Audited Transcripts"])
    # generated_label = trans_annot.tagger()

    # Merge the Rectifier and Label Annotation components output.
    # generated_rect.update(generated_label)

    return generated_rect
