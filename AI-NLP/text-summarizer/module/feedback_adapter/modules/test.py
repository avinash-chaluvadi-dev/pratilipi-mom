import re

# The SOT and EOT tag remover
# test = "[SOT][]yeah[/] [-]we are left with[/-] a week to complete <>a[#]our[/#]</> sprint []yeah[/][EOT]{}"
# start = test.find("[SOT]") + len("[SOT]")
# end = test.find("[EOT]")
# substring = test[start:end]
# tag = "[]"
# print(tag in test)

transcript = "[SOT][]Oh,[/] okay. So that []if[/] people []don't know.[/][EOT]{'Sentiment': 'Neu','Entity': {'Words':  ['people'], 'Type': ['Team Name']},'Deadline': 'Yes','Escalation': 'Yes','Help': 'No','Question': 'No'}"

start = transcript.find("{")
end = transcript.rfind("}") + 1
transcript = transcript[start:end]
print(transcript)
