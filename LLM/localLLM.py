import requests
import json
from retry import retry
import os
from vllm import LLM, SamplingParams
from typing import List, Dict, Union


@retry(tries=3, delay=1)
def VllmLocalLLM(
    messages: List[Dict[str, str]],
    model: str = "/data2/share/llama3.1/llama-3.1-8B-Instruct",
    temperature: float = 0.5,
    top_p: float = 1,
    max_tokens: int = 1024,
    api_key: str = "api_key",
    api_port: int = 8000,
) -> Union[str, Dict[str, str]]:
    """
    Sends a request to a local LLM server to generate a response based on the provided messages.

    Args:
        messages (List[Dict[str, str]]): A list of messages to send to the LLM.
        model (str): The model to use for the LLM.
        temperature (float): The temperature parameter for sampling.
        top_p (float): The top_p parameter for nucleus sampling.
        max_tokens (int): The maximum number of tokens to generate.
        api_key (str): The API key for authentication.
        api_port (int): The port on which the LLM server is running.

    Returns:
        Union[str, Dict[str, str]]: The generated response or an error message.
    """
    url = f"http://localhost:{api_port}/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    data = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return {"error": str(e)}
    except (KeyError, IndexError) as e:
        print(f"Response parsing failed: {e}")
        return {"error": "Invalid response format"}


def set_neuron_environment_variables() -> None:
    """
    Sets environment variables for neuron quantization.
    """
    os.environ["NEURON_CONTEXT_LENGTH_BUCKETS"] = "128,512,1024,2048"
    os.environ["NEURON_TOKEN_GEN_BUCKETS"] = "128,512,1024,2048"
    os.environ["NEURON_QUANT_DTYPE"] = "s8"


def create_llm(
    model: str, device: str, quantization: Union[str, None], tensor_parallel_size: int
) -> LLM:
    """
    Creates an LLM instance with the specified parameters.

    Args:
        model (str): The model to use for the LLM.
        device (str): The device to run the LLM on.
        quantization (Union[str, None]): The quantization method to use.
        tensor_parallel_size (int): The size of the tensor parallelism.

    Returns:
        LLM: The created LLM instance.
    """
    return LLM(
        model=model,
        max_num_seqs=1,
        max_model_len=2048,
        block_size=2048,
        device=device,
        quantization=quantization,
        tensor_parallel_size=tensor_parallel_size,
    )


def prepare_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Prepares a prompt from a list of messages.

    Args:
        messages (List[Dict[str, str]]): A list of messages to prepare the prompt from.

    Returns:
        str: The prepared prompt.
    """
    return "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])


def VllmOfflineLLM(
    messages: List[Dict[str, str]],
    model: str = "/data2/share/llama3.1/llama-3.1-8B-Instruct",
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_tokens: int = 1024,
    quantization: Union[str, None] = None,
    dtype: Union[str, None] = None,
    device: str = "cuda",
    tensor_parallel_size: int = 1,
) -> Union[str, Dict[str, str]]:
    """
    Generates a response from an offline LLM based on the provided messages.

    Args:
        messages (List[Dict[str, str]]): A list of messages to send to the LLM.
        model (str): The model to use for the LLM.
        temperature (float): The temperature parameter for sampling.
        top_p (float): The top_p parameter for nucleus sampling.
        max_tokens (int): The maximum number of tokens to generate.
        quantization (Union[str, None]): The quantization method to use.
        dtype (Union[str, None]): The data type to use for the model.
        device (str): The device to run the LLM on.
        tensor_parallel_size (int): The size of the tensor parallelism.

    Returns:
        Union[str, Dict[str, str]]: The generated response or an error message.
    """
    if quantization == "neuron_quant":
        set_neuron_environment_variables()

    sampling_params = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_tokens
    )
    llm = create_llm(model, device, quantization, tensor_parallel_size)

    if dtype:
        llm.model.to(dtype=dtype)

    prompt = prepare_prompt(messages)
    outputs = llm.generate([prompt], sampling_params)

    if outputs and outputs[0].outputs:
        return outputs[0].outputs[0].text
    else:
        return {"error": "No output generated"}


if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Why is the sky blue?"},
    ]
    response = VllmOfflineLLM(messages=messages)
    print(response)
