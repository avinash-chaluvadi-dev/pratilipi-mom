import logging

import boto3
from botocore import exceptions

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def test_handler(event, context):
    # Initializes step-function client to interact with AWS state machine
    client = boto3.client("stepfunctions")

    # Call start_execution method to invoke state machine
    try:
        response = client.stop_execution(
            executionArn="arn:aws:states:us-east-1:474939635936:execution:mom-state-machine-sit:pratilipi_mom_execution_11-34-01-11-2022_972819b9cb3e40c89e80b46dc22b089c"
        )
    except exceptions.ClientError as error:
        logger.error(error.response["Error"]["Message"])
        raise RuntimeError(error)
