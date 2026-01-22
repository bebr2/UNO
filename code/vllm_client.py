import asyncio
import aiohttp
import json
import logging
from tqdm.asyncio import tqdm_asyncio
from asyncio import as_completed
from typing import List, Dict, Any, Optional
import requests
import os
import subprocess

NEED_COPY_CKPT = os.getenv("NEED_COPY_CKPT", "False") == "True"
LLM_PATH = os.getenv("LLM_PATH", "../LLM")

print(f"NEED_COPY_CKPT: {NEED_COPY_CKPT}")
print(f"LLM_PATH: {LLM_PATH}")

from openai import OpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class VllmAsyncClient:
    DEFAULT_GENERATE_CONFIG = {
        "temperature": 0.1,
        "max_tokens": 2048,
        "seed": 42,
    }

    def __init__(
        self,
        api_url: str = "http://localhost:8000/v1/chat/completions",
        base_model_name: str = None,
        concurrent_requests: int = 10,
        timeout: int = 1800,
        max_retries: int = 3,
        retry_delay: int = 5
    ):
        self.api_url = api_url
        self.base_model_name = base_model_name if base_model_name else "Qwen3-8B"
        self.concurrent_requests = concurrent_requests
        self.timeout_config = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=self.api_url.split("/v1")[0] + "/v1",
        )
        logging.info(f"VllmAsyncClient initialized: URL={self.api_url}, Base Model={self.base_model_name}, Concurrency={self.concurrent_requests}")

    async def generate_batch(
        self,
        questions_data: List[Dict[str, Any]],
        lora_name: Optional[str] = None,
        json_format: Any = None,
        generation_config_overrides: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        model_to_use = self.base_model_name if lora_name is None else lora_name

        final_gen_config = self.DEFAULT_GENERATE_CONFIG.copy()
        if generation_config_overrides:
            final_gen_config.update(generation_config_overrides)

        logging.info(f"Start processing {len(questions_data)} requests... Model: {model_to_use}")

        tasks = []
        sem = asyncio.Semaphore(self.concurrent_requests)

        async with aiohttp.ClientSession(timeout=self.timeout_config) as session:
            with tqdm_asyncio(total=len(questions_data), desc=f"Model: {model_to_use}") as pbar:

                async def limited_task_executor(info):
                    async with sem:
                        return await self._send_request_async(
                            session, info, pbar, model_to_use, json_format, final_gen_config
                        )

                tasks = [limited_task_executor(q_info) for q_info in questions_data]

                results = []
                for coro in as_completed(tasks):
                    try:
                        res = await coro
                        results.append(res)
                    except Exception as e:
                        logging.exception(f"Unhandled exception in task execution: {e}")
                        results.append({"error_in_coro": str(e)})

        if all("id" in result.keys() for result in results):
            return sorted(results, key=lambda x: x["id"])
        elif all("test_idx" in result.keys() for result in results):
            return sorted(results, key=lambda x: x["test_idx"])
        else:
            return results

    async def _send_request_async(
        self,
        session: aiohttp.ClientSession,
        info: Dict[str, Any],
        pbar: tqdm_asyncio,
        model_name: str,
        json_format: Any,
        gen_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        prompt = info["prompt"]
        if type(prompt) is str:
            prompt = [{"role": "user", "content": prompt}]

        headers = {"Content-Type": "application/json"}

        payload = {
            "model": model_name,
            "messages": prompt,
        }
        payload.update(gen_config)

        if json_format is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response-description",
                    "schema": json_format
                },
            }

        last_exception = None
        return_dict = info.copy()

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    **payload
                )
                content_str = response.choices[0].message.content

                if json_format is not None:
                    try:
                        return_dict["_raw_response"] = json.loads(content_str)
                        pbar.update(1)
                        return return_dict
                    except json.JSONDecodeError:
                        logging.error(f"Failed to parse model JSON: {content_str}")
                        logging.warning(f"Request failed (Attempt {attempt + 1}): response content is not valid JSON.")
                        last_exception = "JSONDecodeError"
                else:
                    content_str = response.choices[0].message.content

                    pbar.update(1)
                    return_dict["_raw_response"] = content_str
                    return return_dict

            except Exception as e:
                last_exception = str(e)
                logging.warning(f"Request exception (Attempt {attempt + 1}): {e}")
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay)

        pbar.update(1)
        error_msg = f"error: Max retries ({self.max_retries}) exceeded. Last error: {last_exception}"
        return_dict["_raw_response"] = error_msg
        return return_dict


def load_one_lora(
    lora_path_dir: str,
    lora_name: str,
    vllm_host: str = "http://localhost:8000"
) -> bool:
    url = f"{vllm_host}/v1/load_lora_adapter"
    headers = {"Content-Type": "application/json"}

    if not os.path.exists(lora_path_dir):
        logging.error(f"[!] Path does not exist: {lora_path_dir}")
        return False

    if not os.path.isdir(lora_path_dir):
        logging.error(f"[!] Path is not a directory: {lora_path_dir}")
        return False

    adapter_file_safetensors = os.path.join(lora_path_dir, "adapter_model.safetensors")
    adapter_file_bin = os.path.join(lora_path_dir, "adapter_model.bin")
    if not os.path.exists(adapter_file_safetensors) and not os.path.exists(adapter_file_bin):
        logging.warning(
            f"[!] Warning: adapter_model.safetensors or adapter_model.bin not found in {lora_path_dir}."
        )

    if NEED_COPY_CKPT:
        abs_source_path = os.path.abspath(lora_path_dir)
        relative_path_structure = abs_source_path.lstrip(os.sep)
        target_lora_path_dir = os.path.join(LLM_PATH, relative_path_structure)
        if not os.path.exists(target_lora_path_dir):
            os.makedirs(target_lora_path_dir, exist_ok=True)
            subprocess.run(f"cp -r {lora_path_dir}/* {target_lora_path_dir}", shell=True, check=True)
        logging.info(f"[*] Copy completed: {lora_path_dir} -> {target_lora_path_dir}")
        lora_path_dir = target_lora_path_dir

    if not lora_name:
        logging.error(f"[!] Failed to extract a valid lora_name from path {lora_path_dir}.")
        return False

    payload = {
        "lora_name": lora_name,
        "lora_path": lora_path_dir
    }

    logging.info(f"[*] Preparing to load: {lora_name} (Path: {lora_path_dir})")
    print("-" * 40)

    max_tries = 5
    for j in range(max_tries):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)

            if response.status_code == 200:
                logging.info(f"[+] Success: {lora_name} loaded.")
                print("-" * 40)
                return True
            else:
                logging.error(f"[!] Failed: {lora_name} load failed. Status: {response.status_code}, Resp: {response.text}")
                print("-" * 40)
                print("Retrying")

        except requests.exceptions.RequestException as e:
            logging.error(f"[!] Error: Exception when requesting vLLM server: {e}, LoRA path: {lora_path_dir}")
            print("-" * 40)
            print("Retrying")

    if j == max_tries - 1:
        logging.error(f"[!] Error: {lora_name} load failed after {max_tries} attempts.")
        return False


def load_all_lora_adapters(
    lora_path_dir: str,
    vllm_host: str = "http://localhost:8000"
) -> List[str]:
    url = f"{vllm_host}/v1/load_lora_adapter"
    headers = {"Content-Type": "application/json"}

    checkpoints = []
    if not os.path.exists(lora_path_dir):
        logging.warning(f"LoRA path does not exist: {lora_path_dir}")
        return []

    for item_name in os.listdir(lora_path_dir):
        adapter_file = os.path.join(lora_path_dir, item_name, "adapter_model.safetensors")
        if os.path.exists(adapter_file):
            checkpoints.append(item_name)

    logging.info(f"Found {len(checkpoints)} LoRA adapters.")
    print("-" * 40)

    success_loras = []
    for lora_name in sorted(checkpoints):
        lora_full_path = os.path.join(lora_path_dir, lora_name)

        if NEED_COPY_CKPT:
            abs_source_path = os.path.abspath(lora_full_path)
            relative_path_structure = abs_source_path.lstrip(os.sep)
            target_lora_path_dir = os.path.join(LLM_PATH, relative_path_structure)
            if not os.path.exists(target_lora_path_dir):
                os.makedirs(target_lora_path_dir, exist_ok=True)
                subprocess.run(f"cp -r {lora_full_path}/* {target_lora_path_dir}", shell=True, check=True)
            logging.info(f"[*] Copy completed: {lora_full_path} -> {target_lora_path_dir}")
            lora_full_path = target_lora_path_dir

        payload = {
            "lora_name": lora_name,
            "lora_path": lora_full_path
        }

        logging.info(f"[*] Preparing to load: {lora_name} (Path: {lora_full_path})")
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
            if response.status_code == 200:
                success_loras.append(lora_name)
                logging.info(f"[+] Success: {lora_name} loaded.")
            else:
                logging.error(f"[!] Failed: {lora_name} load failed. Status: {response.status_code}, Resp: {response.text}")
        except requests.exceptions.RequestException as e:
            logging.error(f"[!] Error: Exception when requesting vLLM server: {e}")
            return success_loras
        print("-" * 40)

    return success_loras


def unload_all_lora_adapters(
    lora_names: List[str],
    vllm_host: str = "http://localhost:8000"
):
    logging.info("\n================ Preparing to unload LoRA adapters ================")
    url = f"{vllm_host}/v1/unload_lora_adapter"
    headers = {"Content-Type": "application/json"}

    for lora_name in lora_names:
        payload = {"lora_name": lora_name}
        logging.info(f"[*] Preparing to unload: {lora_name}")
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
            if response.status_code == 200:
                logging.info(f"[+] Success: {lora_name} unloaded.")
            else:
                logging.warning(f"[!] Failed: {lora_name} unload failed. Status: {response.status_code}, Resp: {response.text}")
        except requests.exceptions.RequestException as e:
            logging.error(f"[!] Error: Exception when requesting vLLM server: {e}")
        print("-" * 40)
    logging.info("================ Unload completed ================")

