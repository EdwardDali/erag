import json
from tkinter import filedialog

def upload_json():
    file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
    if file_path:
        with open(file_path, 'r', encoding="utf-8") as json_file:
            data = json.load(json_file)
            return json.dumps(data, ensure_ascii=False)
    return None
