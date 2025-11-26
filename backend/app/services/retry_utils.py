"""
Retry utility for OpenAI API calls with exponential backoff.
Handles transient failures, rate limits, and network issues.
"""
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
    after_log
)
from openai import APIError, RateLimitError, APIConnectionError, APITimeoutError, APIStatusError
import logging
from typing import Callable, TypeVar, Any

logger = logging.getLogger(__name__)

T = TypeVar('T')


def is_retryable_error(exception: Exception) -> bool:
    """
    Determine if an exception is retryable.

    Retries on:
    - RateLimitError (429)
    - APIConnectionError (network issues)
    - APITimeoutError (timeout)
    - Server errors (500, 502, 503, 504)

    Does not retry on:
    - Authentication errors (401)
    - Permission errors (403)
    - Invalid request errors (400)
    - Not found errors (404)
    """
    # OpenAI specific errors
    if isinstance(exception, RateLimitError):
        return True
    if isinstance(exception, APIConnectionError):
        return True
    if isinstance(exception, APITimeoutError):
        return True

    # Check for HTTP status codes in APIError/APIStatusError
    if isinstance(exception, APIStatusError):
        status_code = getattr(exception, 'status_code', None)
        if status_code:
            # Rate limit errors (429)
            if status_code == 429:
                return True
            # Server errors (500, 502, 503, 504)
            if status_code in [500, 502, 503, 504]:
                return True

    # Check for other APIError types
    if isinstance(exception, APIError):
        # Connection errors
        if hasattr(exception, 'code') and exception.code in ['connection_error', 'timeout']:
            return True

    # Network-related exceptions
    if isinstance(exception, (ConnectionError, TimeoutError)):
        return True

    # Check error message for common retryable patterns
    error_str = str(exception).lower()
    retryable_patterns = [
        'rate limit',
        'timeout',
        'connection',
        'network',
        'temporary',
        'server error',
        'internal error',
        'service unavailable',
        'bad gateway',
        'gateway timeout'
    ]

    if any(pattern in error_str for pattern in retryable_patterns):
        return True

    return False


def openai_retry_decorator(
    max_attempts: int = 5,
    initial_wait: float = 1.0,
    max_wait: float = 60.0,
    exponential_base: float = 2.0
):
    """
    Decorator for OpenAI API calls with exponential backoff retry logic.

    Args:
        max_attempts: Maximum number of retry attempts (default: 5)
        initial_wait: Initial wait time in seconds (default: 1.0)
        max_wait: Maximum wait time in seconds (default: 60.0)
        exponential_base: Base for exponential backoff (default: 2.0)

    Returns:
        Decorator function
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=initial_wait,
            min=initial_wait,
            max=max_wait,
            exp_base=exponential_base
        ),
        retry=retry_if_exception(is_retryable_error),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.INFO),
        reraise=True
    )


def retry_openai_call(
    func: Callable[[], T],
    max_attempts: int = 5,
    initial_wait: float = 1.0,
    max_wait: float = 60.0,
    exponential_base: float = 2.0
) -> T:
    """
    Execute an OpenAI API call with retry logic.

    Args:
        func: Function to execute (should be a callable that returns the result)
        max_attempts: Maximum number of retry attempts (default: 5)
        initial_wait: Initial wait time in seconds (default: 1.0)
        max_wait: Maximum wait time in seconds (default: 60.0)
        exponential_base: Base for exponential backoff (default: 2.0)

    Returns:
        Result of the function call

    Raises:
        The last exception if all retries are exhausted
    """
    @openai_retry_decorator(
        max_attempts=max_attempts,
        initial_wait=initial_wait,
        max_wait=max_wait,
        exponential_base=exponential_base
    )
    def _retry_wrapper():
        return func()

    return _retry_wrapper()

