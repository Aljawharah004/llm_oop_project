from abc import ABC, abstractmethod
from tokenizer import Tokenizer
from exceptions import InvalidPromptError

class LanguageModel (ABC):
    total_models = 0
    
    def __init__(self, model_name, provider, max_tokens, temperature, pricing_plan):
        self.model_name = model_name
        self.provider = provider 
        self.max_tokens = max_tokens
        self.temperature = temperature 
        LanguageModel.total_models += 1
        self.tokenizer = Tokenizer()  # Initialize the tokenizer instance (COMPOSITION)
        self.pricing_plan = pricing_plan # Association with PricingPlan (AGGREGATION) 
        self.__is_active = True

    @abstractmethod
    def generate_response(self, prompt):
        pass

    def display_info(self):
        print(f"Model Name: {self.model_name} \nProvider: {self.provider} \nMaximum Tokens: {self.max_tokens} \nTemperature: {self.temperature}")

    @classmethod
    def get_total_models(cls):
        return cls.total_models 

    @staticmethod
    def validate_temperature(temp):
        return True if temp >= 0 and temp <= 2.0 else False 

    @property # Getter for temperature 
    def temperature(self):
        return self.__temperature 

    @temperature.setter # Setter for temperature 
    def temperature(self, value):
        if self.validate_temperature(value):
            self.__temperature = value
        else:
            print("Temperature must be between 0 and 2.") 
    
    @property
    def is_active(self):
        return self.__is_active

    @is_active.setter
    def is_active(self, value):
        self.__is_active = bool(value)

    def calculate_request_cost(self, prompt):
        tokens = self.tokenizer.count_tokens(prompt)
        cost = self.pricing_plan.calculate_cost(tokens)
        print(f"Cost for processing the prompt: ${cost:.4f}")

    # magic methods:
    def __str__(self):
        return f"{self.model_name} by {self.provider}"

    def __repr__(self):
        return f"{type(self).__name__}(model_name='{self.model_name}', provider='{self.provider}')"

    def __len__(self):
        return self.max_tokens 
    
    def __eq__(self, other):
        if not isinstance(other, LanguageModel):
            return NotImplemented
        return self.model_name == other.model_name and self.provider == other.provider 

class LoggingMixin:
    def log(self, message):
        print(f"[LOG] {message}")

class GPTModel(LanguageModel, LoggingMixin):
    def __init__(self, model_name, provider, max_tokens, temperature, api_key, pricing_plan):
        super().__init__(model_name, provider, max_tokens, temperature, pricing_plan)
        self.api_key = api_key 

    def generate_response(self, prompt):
        if not isinstance(prompt, str) or not prompt.strip():
            raise InvalidPromptError("Prompt must be a non-empty string.")
        self.log(f"Sending request to {self.model_name}") # polymorphic behavior from LoggingMixin 
        tokens = self.tokenizer.count_tokens(prompt)
        return f"GPT API response to: {prompt}\nInput tokens: {tokens}"

class LlamaModel(LanguageModel):
    def __init__(self, model_name, provider, max_tokens, temperature, model_path, quantization, pricing_plan):
        super().__init__(model_name, provider, max_tokens, temperature, pricing_plan)
        self.model_path = model_path
        self.quantization = quantization 

    def generate_response(self, prompt):
        if not isinstance(prompt, str) or not prompt.strip():
            raise InvalidPromptError("Prompt must be a non-empty string.")
        tokens = self.tokenizer.count_tokens(prompt)
        return f"Local Llama response to: {prompt}\nInput tokens: {tokens}"
