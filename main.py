from models import GPTModel, LanguageModel , LlamaModel
from pricing import PricingPlan
from tokenizer import Tokenizer
from exceptions import InvalidPromptError
from conversation import Conversation
from manager import ModelManager


print("========== 1. Object Creation ==========")
# Pricing plans
standard_plan = PricingPlan("Standard", 0.01) 
gold_plan = PricingPlan("Gold", 0.015)
premium_plan = PricingPlan("Premium", 0.02)


gpt = GPTModel("GPT-3", "OpenAI", 4096, 0.7, "######", pricing_plan=standard_plan) 
llama = LlamaModel("LLaMA-2", "Meta", 8192, 0.8, "/path/to/llama/model", "int8", pricing_plan=premium_plan)

manager = ModelManager()
manager.add_model(gpt)
manager.add_model(llama)
conversation = Conversation(conversation_id = 1, model = gpt)

print(f"Total models created: {LanguageModel.get_total_models()}")
print("Token count for 'Hello world':", Tokenizer().count_tokens("Hello world"))

print("\n========== 2. Model Execution ==========")
gpt.display_info()
print("\n",repr(gpt))
print(str(gpt)) 
print(f"Max tokens for {gpt.model_name}: {gpt.max_tokens}")

gpt_copy = GPTModel("GPT-3", "OpenAI", 4096, 0.7, "######", pricing_plan=standard_plan)
print("Gpt == Gpt_copy:", gpt == gpt_copy)  
print("Gpt == Llama:", gpt == llama)
print(gpt.generate_response("What is Python?"))
print(llama.generate_response("Explain the concept of machine learning."))

print("\n========== 3. Polymorphism ==========")
models = [gpt, llama]
for model in models:
    print(f"Response from {model.model_name}: {model.generate_response('Tell me a joke.')}")

print("\n========== 4. Encapsulation ==========")
gpt.temperature = 1.5
print(f"Updated temperature for {gpt.model_name}: {gpt.temperature}") 
gpt.temperature = 2.5
print(f"Temperature for {gpt.model_name} after invalid update: {gpt.temperature}")
print(f"Result of validate_temperature(1.0): {LanguageModel.validate_temperature(1.0)}")
print(f"Result of validate_temperature(3.0): {LanguageModel.validate_temperature(3.0)}") 

print("\n========== 5. Abstraction & Exception Handling ==========")

try:
    LanguageModel("Basic-LLM", "Local", 2048, 0.7, pricing_plan=standard_plan)  # This should raise an error
except TypeError as error:
    print("Cannot create an abstract class:", error)

test_prompts = ["", "     ", 123, None, "Explain machine learning"]
for prompt in test_prompts:
    try:
        print(gpt.generate_response(prompt))
    except InvalidPromptError as error:
        print(f"InvalidPromptError for {prompt!r}: {error}")

print("\n========== 6. Conversation Management ==========")
conversation.send_message("Hello, how are you?")
conversation.send_message("Can you explain the concept of AI?") 
conversation.display_history()

print ("\n========== 7. Cost Request ==========")
gpt.calculate_request_cost("What is the capital of France?")
llama.calculate_request_cost("Explain the concept of machine learning.")

print("\n========== 8. Model Manager ==========") 
print("Models in the manager:")
manager.list_models()

print("Selecting a model for provider 'OpenAI' with at least 3000 tokens:")
selected_model = manager.select_model("OpenAI", 3000)
if selected_model:
    print(f"Selected model: {selected_model.model_name}")
else:
    print("No suitable model found.")

print("Selecting with too many tokens:")
print(manager.select_model("Meta", 100000))

llama.is_active = False
print("After deactivating LLaMA-2:", manager.select_model("Meta", 1000))

print("Getting model 'LLaMA-2':", manager.get_model(llama.model_name))

print("removing model 'GPT-3' and listing models again:")
manager.remove_model(gpt.model_name) 
manager.list_models()