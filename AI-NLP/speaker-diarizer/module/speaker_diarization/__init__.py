from . import config
from .ml_engine import engine

"""
Highest Level functions, which will be used to interact with the the package. Current exposed component from the
package is serving component.

"""


def spk_diarize_serve(input_file):
    diarizer = engine.DiarizeEngine()
    response = diarizer.serve(audio_file=input_file, mode=config.RUN_MODE)
    return response
