from __future__ import annotations

import logging
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from langchain_community.utils.openai import is_openai_v1
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.utils import get_from_dict_or_env, get_pydantic_field_names
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from chatchat.server.utils import run_in_thread_pool

logger = logging.getLogger(__name__)


def _create_retry_decorator(embeddings: LocalAIEmbeddings) -> Callable[[Any], Any]:
    import openai

    min_seconds = 4
    max_seconds = 10
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(embeddings.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
            retry_if_exception_type(openai.Timeout)
            | retry_if_exception_type(openai.APIError)
            | retry_if_exception_type(openai.APIConnectionError)
            | retry_if_exception_type(openai.RateLimitError)
            | retry_if_exception_type(openai.InternalServerError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def _async_retry_decorator(embeddings: LocalAIEmbeddings) -> Any:
    import openai

    min_seconds = 4
    max_seconds = 10
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    async_retrying = AsyncRetrying(
        reraise=True,
        stop=stop_after_attempt(embeddings.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
            retry_if_exception_type(openai.Timeout)
            | retry_if_exception_type(openai.APIError)
            | retry_if_exception_type(openai.APIConnectionError)
            | retry_if_exception_type(openai.RateLimitError)
            | retry_if_exception_type(openai.InternalServerError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )

    def wrap(func: Callable) -> Callable:
        async def wrapped_f(*args: Any, **kwargs: Any) -> Callable:
            async for _ in async_retrying:
                return await func(*args, **kwargs)
            raise AssertionError("this is unreachable")

        return wrapped_f

    return wrap


# https://stackoverflow.com/questions/76469415/getting-embeddings-of-length-1-from-langchain-openaiembeddings
def _check_response(response: Any) -> Any:
    # 检查 response 是否是字符串（错误消息）
    if isinstance(response, str):
        import openai
        raise openai.APIError(f"API returned error string: {response}")
    
    # 检查 response 是否有 data 属性
    if not hasattr(response, 'data'):
        import openai
        raise openai.APIError(f"Invalid response format: {type(response)}, missing 'data' attribute")
    
    # 检查 data 是否存在且不为空
    if not response.data or len(response.data) == 0:
        import openai
        raise openai.APIError("LocalAI API returned empty data")
    
    if any([len(d.embedding) == 1 for d in response.data]):
        import openai

        raise openai.APIError("LocalAI API returned an empty embedding")
    return response


def embed_with_retry(embeddings: LocalAIEmbeddings, **kwargs: Any) -> Any:
    """Use tenacity to retry the embedding call."""
    retry_decorator = _create_retry_decorator(embeddings)

    @retry_decorator
    def _embed_with_retry(**kwargs: Any) -> Any:
        import openai
        try:
            response = embeddings.client.create(**kwargs)
            
            # 首先检查 response 是否是字符串（错误消息）
            if isinstance(response, str):
                raise openai.APIError(f"API returned error string: {response}")
            
            # 检查 response 是否是 None
            if response is None:
                raise openai.APIError("API returned None response")
            
            # 检查 response 是否有 data 属性
            if not hasattr(response, 'data'):
                # 尝试将 response 转换为字符串以便调试
                response_str = str(response) if response else "None"
                raise openai.APIError(
                    f"Invalid API response format: {type(response)}, "
                    f"missing 'data' attribute. Response: {response_str[:200]}"
                )
            
            # 调用 _check_response 进行进一步验证
            try:
                return _check_response(response)
            except AttributeError as e:
                # 如果 _check_response 中访问 data 属性失败，提供更详细的错误信息
                if "'str' object has no attribute" in str(e) or "'NoneType' object has no attribute" in str(e):
                    response_str = str(response) if response else "None"
                    raise openai.APIError(
                        f"Invalid API response format: {type(response)}. "
                        f"Response content: {response_str[:200]}. "
                        f"Original error: {e}"
                    )
                raise
        except openai.APIError:
            # 重新抛出 APIError
            raise
        except AttributeError as e:
            # 捕获属性访问错误
            if "'str' object has no attribute" in str(e):
                import openai
                raise openai.APIError(
                    f"API returned invalid response format (string instead of object): {e}"
                )
            raise
        except Exception as e:
            # 其他所有异常
            import openai
            error_type = type(e).__name__
            error_msg = str(e)
            # 如果是字符串响应导致的错误，提供更清晰的错误信息
            if "'str' object has no attribute" in error_msg:
                raise openai.APIError(
                    f"Invalid API response format: API returned string instead of object. "
                    f"Error: {error_type}: {error_msg}"
                )
            # 重新抛出原始异常，让 retry 机制处理
            raise

    return _embed_with_retry(**kwargs)


async def async_embed_with_retry(embeddings: LocalAIEmbeddings, **kwargs: Any) -> Any:
    """Use tenacity to retry the embedding call."""

    @_async_retry_decorator(embeddings)
    async def _async_embed_with_retry(**kwargs: Any) -> Any:
        import openai
        try:
            response = await embeddings.async_client.create(**kwargs)
            
            # 首先检查 response 是否是字符串（错误消息）
            if isinstance(response, str):
                raise openai.APIError(f"API returned error string: {response}")
            
            # 检查 response 是否是 None
            if response is None:
                raise openai.APIError("API returned None response")
            
            # 检查 response 是否有 data 属性
            if not hasattr(response, 'data'):
                # 尝试将 response 转换为字符串以便调试
                response_str = str(response) if response else "None"
                raise openai.APIError(
                    f"Invalid API response format: {type(response)}, "
                    f"missing 'data' attribute. Response: {response_str[:200]}"
                )
            
            # 调用 _check_response 进行进一步验证
            try:
                return _check_response(response)
            except AttributeError as e:
                # 如果 _check_response 中访问 data 属性失败，提供更详细的错误信息
                if "'str' object has no attribute" in str(e) or "'NoneType' object has no attribute" in str(e):
                    response_str = str(response) if response else "None"
                    raise openai.APIError(
                        f"Invalid API response format: {type(response)}. "
                        f"Response content: {response_str[:200]}. "
                        f"Original error: {e}"
                    )
                raise
        except openai.APIError:
            # 重新抛出 APIError
            raise
        except AttributeError as e:
            # 捕获属性访问错误
            if "'str' object has no attribute" in str(e):
                import openai
                raise openai.APIError(
                    f"API returned invalid response format (string instead of object): {e}"
                )
            raise
        except Exception as e:
            # 其他所有异常
            import openai
            error_type = type(e).__name__
            error_msg = str(e)
            # 如果是字符串响应导致的错误，提供更清晰的错误信息
            if "'str' object has no attribute" in error_msg:
                raise openai.APIError(
                    f"Invalid API response format: API returned string instead of object. "
                    f"Error: {error_type}: {error_msg}"
                )
            # 重新抛出原始异常，让 retry 机制处理
            raise

    return await _async_embed_with_retry(**kwargs)


class LocalAIEmbeddings(BaseModel, Embeddings):
    """LocalAI embedding models.

    Since LocalAI and OpenAI have 1:1 compatibility between APIs, this class
    uses the ``openai`` Python package's ``openai.Embedding`` as its client.
    Thus, you should have the ``openai`` python package installed, and defeat
    the environment variable ``OPENAI_API_KEY`` by setting to a random string.
    You also need to specify ``OPENAI_API_BASE`` to point to your LocalAI
    service endpoint.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import LocalAIEmbeddings
            openai = LocalAIEmbeddings(
                openai_api_key="random-string",
                openai_api_base="http://localhost:8080"
            )

    """

    client: Any = Field(default=None, exclude=True)  #: :meta private:
    async_client: Any = Field(default=None, exclude=True)  #: :meta private:
    model: str = "text-embedding-ada-002"
    deployment: str = model
    openai_api_version: Optional[str] = None
    openai_api_base: Optional[str] = Field(default=None, alias="base_url")
    # to support explicit proxy for LocalAI
    openai_proxy: Optional[str] = None
    embedding_ctx_length: int = 8191
    """The maximum number of tokens to embed at once."""
    openai_api_key: Optional[str] = Field(default=None, alias="api_key")
    openai_organization: Optional[str] = Field(default=None, alias="organization")
    allowed_special: Union[Literal["all"], Set[str]] = set()
    disallowed_special: Union[Literal["all"], Set[str], Sequence[str]] = "all"
    chunk_size: int = 1000
    """Maximum number of texts to embed in each batch"""
    max_retries: int = 3
    """Maximum number of retries to make when generating."""
    request_timeout: Union[float, Tuple[float, float], Any, None] = Field(
        default=None, alias="timeout"
    )
    """Timeout in seconds for the LocalAI request."""
    headers: Any = None
    show_progress_bar: bool = False
    """Whether to show a progress bar when embedding."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name not in all_required_field_names:
                warnings.warn(
                    f"""WARNING! {field_name} is not default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)

        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )

        values["model_kwargs"] = extra
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["openai_api_key"] = get_from_dict_or_env(
            values, "openai_api_key", "OPENAI_API_KEY"
        )
        values["openai_api_base"] = get_from_dict_or_env(
            values,
            "openai_api_base",
            "OPENAI_API_BASE",
            default="",
        )
        values["openai_proxy"] = get_from_dict_or_env(
            values,
            "openai_proxy",
            "OPENAI_PROXY",
            default="",
        )

        default_api_version = ""
        values["openai_api_version"] = get_from_dict_or_env(
            values,
            "openai_api_version",
            "OPENAI_API_VERSION",
            default=default_api_version,
        )
        values["openai_organization"] = get_from_dict_or_env(
            values,
            "openai_organization",
            "OPENAI_ORGANIZATION",
            default="",
        )
        try:
            import openai

            if is_openai_v1():
                client_params = {
                    "api_key": values["openai_api_key"],
                    "organization": values["openai_organization"],
                    "base_url": values["openai_api_base"],
                    "timeout": values["request_timeout"],
                    "max_retries": values["max_retries"],
                }

                if not values.get("client"):
                    values["client"] = openai.OpenAI(**client_params).embeddings
                if not values.get("async_client"):
                    values["async_client"] = openai.AsyncOpenAI(
                        **client_params
                    ).embeddings
            elif not values.get("client"):
                values["client"] = openai.Embedding
            else:
                pass
        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        return values

    @property
    def _invocation_params(self) -> Dict:
        openai_args = {
            "model": self.model,
            "timeout": self.request_timeout,
            "extra_headers": self.headers,
            **self.model_kwargs,
        }
        if self.openai_proxy:
            import openai

            openai.proxy = {
                "http": self.openai_proxy,
                "https": self.openai_proxy,
            }  # type: ignore[assignment]  # noqa: E501
        return openai_args

    def _embedding_func(self, text: str, *, engine: str) -> List[float]:
        """Call out to LocalAI's embedding endpoint."""
        # handle large input text
        if self.model.endswith("001"):
            # See: https://github.com/openai/openai-python/issues/418#issuecomment-1525939500
            # replace newlines, which can negatively affect performance.
            text = text.replace("\n", " ")
        response = embed_with_retry(
            self,
            input=[text],
            **self._invocation_params,
        )
        # 检查响应是否有效
        if not hasattr(response, 'data') or not response.data or len(response.data) == 0:
            import openai
            raise openai.APIError(f"Invalid embedding response: {response}")
        return response.data[0].embedding

    async def _aembedding_func(self, text: str, *, engine: str) -> List[float]:
        """Call out to LocalAI's embedding endpoint."""
        # handle large input text
        if self.model.endswith("001"):
            # See: https://github.com/openai/openai-python/issues/418#issuecomment-1525939500
            # replace newlines, which can negatively affect performance.
            text = text.replace("\n", " ")
        response = await async_embed_with_retry(
            self,
            input=[text],
            **self._invocation_params,
        )
        # 检查响应是否有效
        if not hasattr(response, 'data') or not response.data or len(response.data) == 0:
            import openai
            raise openai.APIError(f"Invalid embedding response: {response}")
        return response.data[0].embedding

    def embed_documents(
        self, texts: List[str], chunk_size: Optional[int] = 0
    ) -> List[List[float]]:
        """Call out to LocalAI's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.
            chunk_size: The chunk size of embeddings. If None, will use the chunk size
                specified by the class.

        Returns:
            List of embeddings, one for each text.
        """

        # call _embedding_func for each text with multithreads
        def task(seq, text):
            return (seq, self._embedding_func(text, engine=self.deployment))

        params = [{"seq": i, "text": text} for i, text in enumerate(texts)]
        result = list(run_in_thread_pool(func=task, params=params))
        result = sorted(result, key=lambda x: x[0])
        return [x[1] for x in result]

    async def aembed_documents(
        self, texts: List[str], chunk_size: Optional[int] = 0
    ) -> List[List[float]]:
        """Call out to LocalAI's embedding endpoint async for embedding search docs.

        Args:
            texts: The list of texts to embed.
            chunk_size: The chunk size of embeddings. If None, will use the chunk size
                specified by the class.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = []
        for text in texts:
            response = await self._aembedding_func(text, engine=self.deployment)
            embeddings.append(response)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Call out to LocalAI's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        embedding = self._embedding_func(text, engine=self.deployment)
        return embedding

    async def aembed_query(self, text: str) -> List[float]:
        """Call out to LocalAI's embedding endpoint async for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        embedding = await self._aembedding_func(text, engine=self.deployment)
        return embedding
