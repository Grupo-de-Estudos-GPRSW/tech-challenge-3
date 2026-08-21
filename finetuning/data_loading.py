import os
import pandas as pd
import xml.etree.ElementTree as ET

import subprocess
from pathlib import Path

medquad_dir = Path("MedQuAD")

if not medquad_dir.is_dir():
    print("Pasta MedQuAD não encontrada. Clonando repositório...")
    subprocess.run(["git", "clone", "https://github.com/abachaa/MedQuAD.git"], check=True)
    print("Repositório clonado")
else:
    print("Pasta MedQuAD já existe, não é necessário clonar novamente.")

def parse_file(file_path):
    tree = ET.parse(file_path)
    qapairs = tree.iter("QAPair")

    parsed_data = []

    for qapair in qapairs:
        question_elem = qapair.find("Question")
        answer_elem = qapair.find("Answer")

        if question_elem is not None and answer_elem is not None:
            question = question_elem.text
            answer = answer_elem.text

            if question is not None and answer is not None:
                parsed_data.append({"question": question, "answer": answer})
    return parsed_data

docs = []

for folder in os.listdir(medquad_dir):
    folder_path = medquad_dir / folder
    if folder_path.is_dir():
        for file in os.listdir(folder_path):
            if file.endswith(".xml"):
                file_path = folder_path / file
                docs.extend(parse_file(file_path))

df = pd.DataFrame(docs)
