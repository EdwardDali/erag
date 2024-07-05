import subprocess

def get_available_models(api_type):
    if api_type == "ollama":
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            models = result.stdout.strip().split('\n')[1:]  # Skip the header
            # Filter out 'failed' and 'NAME' entries
            return [model.split()[0] for model in models if model.split()[0] not in ['failed', 'NAME']]
        except subprocess.CalledProcessError:
            print("Error running 'ollama list' command")
            return []
    elif api_type == "llama":
        # TODO: Implement model listing for llama API
        return ["llama-default"]  # Placeholder
    else:
        return []

def update_settings(settings, api_type, model):
    if api_type == "ollama":
        settings.update_setting("ollama_model", model)
    elif api_type == "llama":
        # TODO: Update settings for llama API
        pass
    settings.apply_settings()
    print(f"Settings updated. Using {model} with {api_type} API.")
