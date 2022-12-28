import asyncio
import logging as lg

import aiohttp
import requests

from rest_api.models import File

from .utils import update_ml_execution_status_in_db

logger = lg.getLogger("file")


def sync_get(url: str):
    """Synchronous Http Get method"""
    try:
        return requests.get(url)
    except requests.exceptions.Timeout:
        logger.error(f"Timeout error while requesting {url}")
    except requests.exceptions.TooManyRedirects:
        logger.error(f"too many redirects while requests=ing {url}")
    except Exception as e:
        logger.error(
            "Unable to get url {} due to {}.".format(
                url, getattr(e, "message", repr(e))
            )
        )


async def get(url: str, session: aiohttp.ClientSession):
    """Helper method for making async http call"""
    try:
        async with session.get(url=url) as response:
            logger.info("Successfully got url {}.".format(url))
        return response
    except Exception as e:
        logger.error(
            "Unable to get url {} due to {}.".format(
                url, getattr(e, "message", repr(e))
            )
        )


@update_ml_execution_status_in_db
async def async_get_requests(urls: list, model_names: list, file: File) -> list:
    """Helper function for making a bunch of async get requests
    Args:
        urls:  list of urls on which async call needs to be done
    """
    async with aiohttp.ClientSession() as session:
        return await asyncio.gather(*[get(url, session) for url in urls])
