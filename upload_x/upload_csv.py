import csv
from tkinter import filedialog

def upload_csv():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        with open(file_path, 'r', encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file)
            text = ""
            for row in csv_reader:
                text += ', '.join(row) + "\n"
            return text
    return None
