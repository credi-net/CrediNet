import os
from openai import OpenAI
from vllm_helpers import _model_to_port



class APIClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = self._create_api_client()

    def _create_api_client(self):
        raise NotImplementedError

    def query_model(self, model, system_prompt, user_instruction):
        raise NotImplementedError


class OpenAIClient(APIClient):
    def __init__(self, api_key=None, model_name=None, enable_web_search=False):
        self.model_name = model_name or "gpt-4o"  # Store model name, default to gpt-4o
        self.enable_web_search = enable_web_search  # Enable web search for queries
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
        super().__init__(api_key)

    def _create_api_client(self):
        return OpenAI(api_key=self.api_key)

    def query_model(self, model, system_prompt, user_instruction):
        """
        Query OpenAI model with optional web search.
        
        Args:
            model: Model name to use
            system_prompt: System prompt
            user_instruction: User instruction
            
        Returns:
            String response from the model
        """
        # Use Responses API for web search, chat.completions for regular queries
        if self.enable_web_search:
            # Combine system prompt and user instruction for responses API
            full_input = f"{system_prompt}\n\n{user_instruction}"
            response = self.client.responses.create(
                model=model,
                tools=[{"type": "web_search"}],
                input=full_input,
            )
            return response.output_text
        else:
            kwargs = {
                "model": model,
                "temperature": 1.0,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_instruction},
                ],
            }
            completion = self.client.chat.completions.create(**kwargs)
            return completion.choices[0].message.content

    def query(self, messages):
        """
        Query OpenAI with a list of messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            String response from the model
        """
        # Use Responses API for web search, chat.completions for regular queries
        if self.enable_web_search:
            # Convert messages to single input string for responses API
            full_input = "\n\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            response = self.client.responses.create(
                model=self.model_name,
                tools=[{"type": "web_search"}],
                input=full_input,
            )
            return response.output_text
        else:
            kwargs = {
                "model": self.model_name,  # Use stored model name
                "temperature": 1.0,
                "messages": messages,
            }
            completion = self.client.chat.completions.create(**kwargs)
            return completion.choices[0].message.content

class VLLMClient(APIClient):
    def __init__(self, api_key=None, base_url=None, model_name=None):
        """
        VLLMClient for querying vLLM servers.
        
        Args:
            api_key: Not used for vLLM (kept for compatibility)
            base_url: Optional explicit base URL. If not provided, will be 
                     determined from model_name using port mapping
            model_name: Model name to determine port (if base_url not provided)
        """
        self.model_name = model_name  # Store model_name for later use
        if base_url is None:
            if model_name is None:
                raise ValueError("Either base_url or model_name must be provided")
            port = _model_to_port(model_name)
            self.base_url = f"http://localhost:{port}/v1"
        else:
            self.base_url = base_url
        super().__init__(api_key)

    def _create_api_client(self):
        return OpenAI(
            api_key="EMPTY",          # vLLM ignores this
            base_url=self.base_url,   # this is the important part
        )

    def query_model(self, model, system_prompt, user_instruction):
        completion = self.client.chat.completions.create(
            model=model,
            temperature=1.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_instruction},
            ],
        )
        return completion.choices[0].message.content

    def query(self, messages):
        """
        Query the vLLM server with a list of messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            String response from the model
        """
        # Use stored model_name if available, otherwise let vLLM use its default
        model = self.model_name if self.model_name else "model"
        
        completion = self.client.chat.completions.create(
            model=model,
            temperature=1.0,
            messages=messages,
        )
        return completion.choices[0].message.content


def create_api_client(provider, api_key=None, model_name=None, base_url=None, enable_web_search=False):
    """
    Create an API client for the specified provider.
    
    Args:
        provider: Provider name ("vllm" or "openai")
        api_key: API key (optional for vllm, required for openai)
        model_name: Model name (used for vllm port mapping or API model selection)
        base_url: Explicit base URL (optional, for vllm)
        enable_web_search: Enable web search for supported models (default: False)
    """
    if provider == "vllm":
        return VLLMClient(api_key=api_key, base_url=base_url, model_name=model_name)
    elif provider == "openai":
        return OpenAIClient(api_key=api_key, model_name=model_name, enable_web_search=enable_web_search)
    else:
        raise ValueError(
            f"Unknown provider: {provider}, must be 'vllm' or 'openai'"
        )