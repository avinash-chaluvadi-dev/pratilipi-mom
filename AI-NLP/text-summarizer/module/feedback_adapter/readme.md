# Annotation Adapter

## Overview

This component cleans up the audited/human reviewed transcripts applying rules on
top of the corresponding transcripts and segregate the annotation labels from the tagged labels.

## Dependencies
The module runs with standard python modules. No Thirds party dependencies are involved.

## How to run 

### Standalone testing.

This mode is used perform the functional testing of the module.
The parameters for the standalone testing can be found in the config.py

`python feedback_adapter/main.py` 

### Dummy api test.
The audited_json and the rules_json paths are needed to be added in the script. Check the script for details.
This script is used to test the api integration of the feedback_adaptor component.

`python api.py`

## Output Storage

In the standalone testing mode, the results are stored in the result folder where the execution 
log and the output JSON with timestamp can be found.

In the api integration mode, the output path can be hardcoded in the api.py script where to store
and check the output.
