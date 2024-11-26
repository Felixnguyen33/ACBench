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


class VllmOfflineModel:
    def __init__(
        self,
        model_path: str = "/data2/share/llama3.1/llama-3.1-8B-Instruct",
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 1024,
        quantization: Union[str, None] = None,
        dtype: Union[str, None] = None,
        device: str = "cuda",
        tensor_parallel_size: int = 1
    ):
        """
        Initialize the VllmOfflineModel with the specified parameters.

        Args:
            model_path (str): Path to the model
            temperature (float): Temperature parameter for sampling
            top_p (float): Top-p parameter for nucleus sampling
            max_tokens (int): Maximum number of tokens to generate
            quantization (Union[str, None]): Quantization method
            dtype (Union[str, None]): Data type for the model
            device (str): Device to run the model on
            tensor_parallel_size (int): Size of tensor parallelism
        """
        self.model_path = model_path
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.quantization = quantization
        self.dtype = dtype
        self.device = device
        self.tensor_parallel_size = tensor_parallel_size
        
        # Initialize the LLM
        if self.quantization == "neuron_quant":
            self._set_neuron_environment()
        
        self.llm = self._create_llm()
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens
        )

    def _set_neuron_environment(self) -> None:
        """Sets environment variables for neuron quantization."""
        os.environ["NEURON_CONTEXT_LENGTH_BUCKETS"] = "128,512,1024,2048"
        os.environ["NEURON_TOKEN_GEN_BUCKETS"] = "128,512,1024,2048"
        os.environ["NEURON_QUANT_DTYPE"] = "s8"

    def _create_llm(self) -> LLM:
        """Creates and returns an LLM instance with the specified parameters."""
        return LLM(
            model=self.model_path,
            max_num_seqs=1,
            max_model_len=2048,
            block_size=2048,
            device=self.device,
            quantization=self.quantization,
            tensor_parallel_size=self.tensor_parallel_size
        )

    def _prepare_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Prepares a prompt from a list of messages.
        
        Args:
            messages (List[Dict[str, str]]): List of message dictionaries
            
        Returns:
            str: Formatted prompt string
        """
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

    def generate(self, messages: List[Dict[str, str]]) -> Union[str, Dict[str, str]]:
        """
        Generates a response based on the input messages.
        
        Args:
            messages (List[Dict[str, str]]): List of input messages
            
        Returns:
            Union[str, Dict[str, str]]: Generated response or error message
        """
        prompt = self._prepare_prompt(messages)
        outputs = self.llm.generate([prompt], self.sampling_params)

        if outputs and outputs[0].outputs:
            return outputs[0].outputs[0].text
        return {"error": "No output generated"}

if __name__ == "__main__":
    # Example usage
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Why is the sky blue?"},
    ]
    
    model = VllmOfflineModel()
    response = model.generate(messages)
    print(response)
