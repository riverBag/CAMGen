import argparse
import asyncio
import json
import os
from types import SimpleNamespace
from openai import AsyncOpenAI
from dotenv import load_dotenv
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from tenacity import RetryError, retry, stop_after_attempt, wait_random_exponential
from urllib import error as urllib_error
from urllib import request as urllib_request

# Load environment variables from .env file
load_dotenv()

# Global OpenAI client, initialized lazily
client = None
api_key_env_name = None
base_url_env_name = None
extra_headers_json = None
transport_mode = "openai"
max_completion_tokens = 1024
thinking_mode = "default"
urllib_info_printed = False


def configure_client(
    api_key_env: str | None = None,
    base_url_env: str | None = None,
    headers_json: str | None = None,
    transport: str = "openai",
    max_tokens: int = 1024,
    thinking: str = "default",
):
    """Configure which environment variables and extra headers the client uses."""
    global client, api_key_env_name, base_url_env_name, extra_headers_json
    global transport_mode, max_completion_tokens, thinking_mode, urllib_info_printed
    client = None
    urllib_info_printed = False
    api_key_env_name = api_key_env
    base_url_env_name = base_url_env
    extra_headers_json = headers_json
    transport_mode = transport
    max_completion_tokens = max_tokens
    thinking_mode = thinking


def _get_env_value(env_names: list[str]) -> tuple[str | None, str | None]:
    for env_name in env_names:
        if not env_name:
            continue
        value = os.getenv(env_name)
        if value:
            value = value.strip()
            if value:
                return value, env_name
    return None, None


def _parse_extra_headers() -> dict[str, str] | None:
    raw_headers = extra_headers_json or os.getenv("OPENAI_EXTRA_HEADERS_JSON")
    if not raw_headers:
        return None

    parsed = json.loads(raw_headers)
    if isinstance(parsed, dict):
        return {str(name): str(value) for name, value in parsed.items()}
    if isinstance(parsed, list):
        headers = {}
        for item in parsed:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            value = item.get("value")
            if name is not None and value is not None:
                headers[str(name)] = str(value)
        return headers
    raise ValueError("Extra headers must be a JSON object or a list of name/value objects.")


def _resolve_api_credentials() -> tuple[str | None, str | None, dict[str, str] | None, str | None, str | None]:
    api_key_envs = [api_key_env_name] if api_key_env_name else [
        "OPENAI_API_KEY",
        "claude_API_KEY",
        "PKU_API_KEY",
    ]
    base_url_envs = [base_url_env_name] if base_url_env_name else [
        "OPENAI_BASE_URL",
        "claude_BASE_URL",
        "PKU_BASE_URL",
    ]
    api_key, resolved_api_key_env = _get_env_value(api_key_envs)
    base_url, resolved_base_url_env = _get_env_value(base_url_envs)
    default_headers = _parse_extra_headers()
    return api_key, base_url, default_headers, resolved_api_key_env, resolved_base_url_env


def _make_openai_like_response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _call_llm_via_urllib(messages: list, model_name: str):
    global urllib_info_printed
    api_key, base_url, default_headers, _, resolved_base_url_env = _resolve_api_credentials()
    if not urllib_info_printed:
        print(f"base_url ({resolved_base_url_env}): {base_url}")
        if default_headers:
            print(f"extra_headers: {', '.join(sorted(default_headers))}")
        urllib_info_printed = True
    if not api_key or not base_url:
        raise ValueError(
            "Configured API key/base URL environment variables are not set. "
            "Please create a .env file and set them, set environment variables, "
            "or pass --api-key-env/--base-url-env."
        )

    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_completion_tokens,
        "temperature": 0,
    }
    if thinking_mode in {"enabled", "disabled"}:
        payload["thinking"] = {"type": thinking_mode}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if default_headers:
        headers.update(default_headers)

    req = urllib_request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
    except urllib_error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body}") from e

    data = json.loads(body)
    content = data["choices"][0]["message"].get("content")
    if not isinstance(content, str):
        content = ""
    return _make_openai_like_response(content)


class AsyncRateLimiter:
    """Spaces request starts across workers to stay below API RPM quotas."""

    def __init__(self, rpm_limit: float):
        self.interval = 60.0 / rpm_limit if rpm_limit and rpm_limit > 0 else 0.0
        self.next_allowed_at = 0.0
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        if self.interval <= 0:
            return
        async with self.lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            if now < self.next_allowed_at:
                await asyncio.sleep(self.next_allowed_at - now)
                now = loop.time()
            self.next_allowed_at = now + self.interval

def get_client():
    """Initializes and returns the OpenAI client, reusing it if already created."""
    global client
    if client is None:
        api_key, base_url, default_headers, _, resolved_base_url_env = _resolve_api_credentials()
        print(f"base_url ({resolved_base_url_env}): {base_url}")
        if default_headers:
            print(f"extra_headers: {', '.join(sorted(default_headers))}")
        #input("Press Enter to continue...")
        if not api_key or not base_url:
            raise ValueError(
                "Configured API key/base URL environment variables are not set. "
                "Please create a .env file and set them, set environment variables, "
                "or pass --api-key-env/--base-url-env."
            )
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
        )
    return client

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
async def call_llm(messages: list, model_name: str):
    """Calls the LLM with exponential backoff retry logic."""
    try:
        if transport_mode == "urllib":
            return await asyncio.to_thread(_call_llm_via_urllib, messages, model_name)
        aclient = get_client()
        response = await aclient.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_completion_tokens,  # Commit messages are relatively short
            temperature=0
        )
        return response
    except Exception as e:
        print(f"Error calling LLM: {e}")
        raise


def classify_failure(error: Exception | None) -> str:
    """Return a compact failure category for exhausted LLM requests."""
    root_error = error
    if isinstance(error, RetryError):
        last_error = error.last_attempt.exception()
        if isinstance(last_error, Exception):
            root_error = last_error

    message = str(root_error or "").lower()
    if any(marker in message for marker in (
        "context length",
        "context_length",
        "maximum context",
        "max context",
        "too many tokens",
        "token limit",
    )):
        return "context_length_exceeded"
    if "rate limit" in message or "rate_limit" in message or "429" in message:
        return "rate_limit"
    if "insufficient balance" in message or "402" in message:
        return "insufficient_balance"
    if "timeout" in message or "timed out" in message:
        return "timeout"
    if any(marker in message for marker in ("authentication", "unauthorized", "invalid api key", "401")):
        return "authentication_error"
    if root_error is None:
        return "api_error"
    return type(root_error).__name__

def make_progress(note: str = "processing"):
    """Creates a rich progress bar."""
    return Progress(
        TextColumn(f"{note} • [progress.percentage]{{task.percentage:>3.0f}}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    )

async def run_task(
    task: dict,
    p: Progress,
    task_id,
    output_file: str,
    failure_output_file: str,
    semaphore: asyncio.Semaphore,
    model_name: str,
    rate_limiter: AsyncRateLimiter | None = None,
):
    """Runs a single task, including the LLM call and writing the result."""
    response_obj = None
    final_error = None

    async with semaphore:
        try:
            if rate_limiter is not None:
                await rate_limiter.wait()
            response_obj = await call_llm(task["messages"], model_name)
        except Exception as e:
            print(f"Failed to get response for task_id {task['task_id']}: {e}")
            response_obj = None
            final_error = e

    p.update(task_id, advance=1)

    failure_type = None
    if response_obj is None:
        failure_type = classify_failure(final_error)
        result = {
            "task_id": task["task_id"],
            "label": task.get("label"),
            "model": model_name,
            "response": None,  # Indicate failure
        }
    else:
        response_content = response_obj.choices[0].message.content
        if not isinstance(response_content, str) or not response_content.strip():
            failure_type = "empty_response"
            response_content = None
        result = {
            "task_id": task["task_id"],
            "label": task.get("label"),
            "model": model_name,
            "response": response_content,
        }

    # Append the result to the output file
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    if failure_type is not None:
        failure_record = {
            "task_id": task["task_id"],
            "failure_type": failure_type,
        }
        with open(failure_output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(failure_record, ensure_ascii=False) + "\n")

    return result

async def run_tasks(
    tasks: list,
    concurrency: int,
    output_file: str,
    failure_output_file: str,
    model_name: str,
    rpm_limit: float = 0.0,
):
    """Runs tasks with a bounded worker pool instead of materializing all coroutines."""
    semaphore = asyncio.Semaphore(concurrency)
    rate_limiter = AsyncRateLimiter(rpm_limit) if rpm_limit and rpm_limit > 0 else None
    queue: asyncio.Queue = asyncio.Queue()
    for task in tasks:
        queue.put_nowait(task)

    async def worker() -> None:
        while True:
            try:
                task = queue.get_nowait()
            except asyncio.QueueEmpty:
                return

            try:
                await run_task(
                    task,
                    p,
                    progress_task_id,
                    output_file,
                    failure_output_file,
                    semaphore,
                    model_name,
                    rate_limiter,
                )
            finally:
                queue.task_done()

    with make_progress("Generating Commits") as p:
        progress_task_id = p.add_task("Querying LLM", total=len(tasks))
        workers = [asyncio.create_task(worker()) for _ in range(max(1, concurrency))]
        await queue.join()
        await asyncio.gather(*workers)

async def main(
    input_file: str,
    output_file: str,
    failure_output_file: str,
    concurrency: int,
    model_name: str,
    rpm_limit: float = 0.0,
):
    """Main function to load tasks and orchestrate the run."""
    # Ensure the output file is cleared before starting
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(failure_output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8"):
        pass
    with open(failure_output_file, "w", encoding="utf-8"):
        pass

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            tasks = [json.loads(line) for line in f]
        print(f"Loaded {len(tasks)} tasks from {input_file}.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return
    
    await run_tasks(tasks, concurrency, output_file, failure_output_file, model_name, rpm_limit)
    print(f"\nProcessing complete. Results saved to {os.path.abspath(output_file)}")
    print(f"Final failures saved to {os.path.abspath(failure_output_file)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch process commit message generation tasks from a JSONL file using an OpenAI-compatible API."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input JSONL file containing tasks (e.g., tasks.jsonl).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output JSONL file to save responses.",
    )
    parser.add_argument(
        "--failure-output",
        default=None,
        help=(
            "Path for requests that still fail after all retries. Defaults to "
            "'<output stem>_failures.jsonl'."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of concurrent tasks (default: 10).",
    )
    parser.add_argument(
        "--rpm-limit",
        type=float,
        default=0.0,
        help="Optional global request start limit per minute across workers. 0 disables throttling.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-v3.2-128k",
        help="The name of the model to use (e.g., deepseek-chat).",
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Environment variable name containing the API key, e.g. PKU_API_KEY.",
    )
    parser.add_argument(
        "--base-url-env",
        default=None,
        help="Environment variable name containing the API base URL, e.g. PKU_BASE_URL.",
    )
    parser.add_argument(
        "--extra-headers-json",
        default=None,
        help="Extra default headers as JSON object or [{'name': ..., 'value': ...}] list.",
    )
    parser.add_argument(
        "--transport",
        choices=["openai", "urllib"],
        default="openai",
        help="HTTP client path to use. urllib avoids AsyncOpenAI and uses raw HTTP.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum completion tokens for each request.",
    )
    parser.add_argument(
        "--thinking",
        choices=["default", "enabled", "disabled"],
        default="default",
        help="Optional DeepSeek thinking mode payload setting.",
    )
    args = parser.parse_args()

    if args.failure_output is None:
        output_root, output_ext = os.path.splitext(args.output)
        output_ext = output_ext or ".jsonl"
        args.failure_output = f"{output_root}_failures{output_ext}"

    configure_client(
        args.api_key_env,
        args.base_url_env,
        args.extra_headers_json,
        args.transport,
        args.max_tokens,
        args.thinking,
    )

    asyncio.run(
        main(
            args.input,
            args.output,
            args.failure_output,
            args.workers,
            args.model,
            args.rpm_limit,
        )
    )
