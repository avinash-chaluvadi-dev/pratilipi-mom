class NoneError(Exception):
    pass


class RequestError(Exception):
    pass


class DownstreamAPIError(Exception):
    pass


class LambdaTimeoutError(Exception):
    pass
