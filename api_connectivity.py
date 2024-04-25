from openai import OpenAI

def configure_api(api_type):
    if api_type == "ollama":
        return OpenAI(
            base_url='http://localhost:11434/v1',
            api_key='phi3:instruct',
             )
    elif api_type == "llama":
        return OpenAI(
            base_url='http://localhost:8080/v1',
            api_key='sk-no-key-required'
        )
    else:
        raise ValueError("Invalid API type")
