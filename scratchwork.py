from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_auth_token="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")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_auth_token="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")