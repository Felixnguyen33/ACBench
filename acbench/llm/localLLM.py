import requests
import json
from retry import retry
import os
import torch
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
        temperature: float = 0,
        top_p: float = 0.95,
        max_tokens: int = 1024,
        # quantization: Union[str, None] = None,
        dtype: Union[str, None] = None,
        device: str = "cuda",
        tensor_parallel_size: int = 1,
        block_size=None,
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
        # Clear GPU memory before initialization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"Cleared GPU memory. Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        self.model_path = model_path
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        # self.quantization = quantization
        self.dtype = dtype
        self.device = device
        
        # Calculate tensor parallel size from CUDA_VISIBLE_DEVICES if not explicitly set
        if tensor_parallel_size == 1 and "CUDA_VISIBLE_DEVICES" in os.environ:
            self.tensor_parallel_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        else:
            self.tensor_parallel_size = tensor_parallel_size
        
        # Initialize the LLM
        # if self.quantization == "neuron_quant":
        #     self._set_neuron_environment()
        
        try:
            self.llm = self._create_llm()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("CUDA out of memory error. Trying with reduced memory settings...")
                # Try with even more conservative settings
                self._create_llm_with_reduced_memory()
            else:
                raise e
        
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop_token_ids=[128009],
            stop=["END", "---", "\n\n"],
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
            trust_remote_code=True,
            max_num_seqs=1,
            max_model_len=8192,  # Increased to handle very long prompts
            # quantization=self.quantization,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=0.7,  # Further reduced from 0.9
            max_num_batched_tokens=1024,  # Further reduced from 2048
            dtype="auto",  # Let vLLM choose optimal dtype
            swap_space=4,  # Add swap space for memory management
            enforce_eager=True,  # Use eager mode to reduce memory usage
        )

    def _create_llm_with_reduced_memory(self) -> LLM:
        """Creates and returns an LLM instance with minimal memory usage."""
        print("Creating LLM with minimal memory settings...")
        return LLM(
            model=self.model_path,
            trust_remote_code=True,
            max_num_seqs=1,
            max_model_len=4096,  # Increased to handle longer prompts
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=0.5,  # Very conservative
            max_num_batched_tokens=512,  # Very conservative
            dtype="fp16",  # Use fp16 to save memory
            swap_space=8,  # More swap space
            enforce_eager=True,
            disable_log_stats=True,  # Disable logging to save memory
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

    def _estimate_prompt_length(self, messages: List[Dict[str, str]]) -> int:
        """
        Estimates the token length of the prompt.
        
        Args:
            messages (List[Dict[str, str]]): List of message dictionaries
            
        Returns:
            int: Estimated token length
        """
        prompt = self._prepare_prompt(messages)
        # Rough estimation: 1 token ≈ 4 characters for English text
        return len(prompt) // 4

    def _adjust_model_len_if_needed(self, messages: List[Dict[str, str]]) -> None:
        """
        Dynamically adjusts max_model_len if the prompt is too long.
        
        Args:
            messages (List[Dict[str, str]]): List of message dictionaries
        """
        estimated_length = self._estimate_prompt_length(messages)
        current_max_len = self.llm.llm_engine.model_config.max_model_len
        
        if estimated_length > current_max_len:
            print(f"Warning: Estimated prompt length ({estimated_length}) exceeds current max_model_len ({current_max_len})")
            print("This might cause issues. Consider increasing max_model_len in the configuration.")

    def generate(self, messages: List[Dict[str, str]]) -> Union[str, Dict[str, str]]:
        """
        Generates a response based on the input messages.
        
        Args:
            messages (List[Dict[str, str]]): List of input messages
            
        Returns:
            Union[str, Dict[str, str]]: Generated response or error message
        """
        # Check if prompt length might be an issue
        self._adjust_model_len_if_needed(messages)
        
        prompt = self._prepare_prompt(messages)
        outputs = self.llm.generate([prompt], self.sampling_params)

        if outputs and outputs[0].outputs:
            return outputs[0].outputs[0].text
        return {"error": "No output generated"}

if __name__ == "__main__":
    # Example usage
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Please answer 'END' when you are done."},
        {"role": "user", "content": "How to make a computer from sand?"},
    ]
    
    model = VllmOfflineModel(
        model_path="/data2/share/llama3.1/llama-3.1-8B-Instruct-awq-w4-g128-zp",
        # -awq-w4-g128-zp",
        # quantization="awq",
        device="cuda"
    )
    response = model.generate(messages)
    print('===================')
    print(response)
    print('===================')
