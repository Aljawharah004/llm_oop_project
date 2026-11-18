from models import LanguageModel


class ModelManager:
    def __init__(self):
        self.models = {}

    def add_model(self, model):
        if isinstance(model, LanguageModel):
            self.models[model.model_name] = model # key : value pair 
        else:
            print("Only instances of LanguageModel can be added.")
            return 

    def remove_model(self, model_name):
        if model_name in self.models:
            del self.models[model_name]
        else:
            print(f"Model {model_name} not found.")

    def get_model(self, model_name):    
        if model_name in self.models:
            return self.models[model_name]
        else:
            print(f"Model {model_name} not found.")
        
    def list_models(self):
        for name in self.models:
            print(name)

    def select_model(self, provider, required_tokens):
        for model in self.models.values():
            if (model.provider == provider and model.max_tokens >= required_tokens and model.is_active):
                return model
        print(f"No active model found for provider {provider} with at least {required_tokens} tokens.")
    
