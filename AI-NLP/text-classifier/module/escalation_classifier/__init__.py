def escalation(data, escalation_list):
    for transcript in data:
        for word in escalation_list:
            if word in transcript["transcript"]:
                transcript["label"] = "Escalation"
                break
            else:
                transcript["label"] = "Others"
    return data


ESCALATION_LIST = ["escalate", "escalation", "escalated"]


def main(data):
    try:
        output = escalation(data["response"], ESCALATION_LIST)
        data["status"] = "success"
        data["model"] = "escalation_classifier"
        data["response"] = output
        return data
    except RuntimeError:
        self.logger.exception(
            "Exception encountered while serving the Escalation Classifier Engine",
            exc_info=True,
        )
        data["status"] = "error"
        data["model"] = "escalation_classifier"
        return data
