import os
import re
import json
from pathlib import Path
from tree_sitter_languages import get_parser, get_language

# TypeScript 파서 준비
ts_language = get_language("typescript")
ts_parser = get_parser("typescript")
ts_parser.set_language(ts_language)

def extract_script_block(vue_code: str) -> str:
    match = re.search(r"<script[^>]*>(.*?)</script>", vue_code, re.DOTALL)
    return match.group(1).strip() if match else ""

def get_method_chunks(script_code: str) -> list:
    tree = ts_parser.parse(bytes(script_code, "utf8"))
    root = tree.root_node

    methods = []
    for node in root.walk():
        if node.type == "class_declaration":
            class_name = ""
            for c in node.named_children:
                if c.type == "identifier":
                    class_name = script_code[c.start_byte:c.end_byte]
                if c.type == "method_definition":
                    start = c.start_point[0]
                    end = c.end_point[0]
                    code_lines = script_code.splitlines()[start:end+1]
                    method_code = "\n".join(code_lines).strip()
                    methods.append({
                        "type": "method",
                        "name": class_name,
                        "content": method_code
                    })
        elif node.type == "function_declaration":
            name_node = node.child_by_field_name("name")
            func_name = script_code[name_node.start_byte:name_node.end_byte]
            start = node.start_point[0]
            end = node.end_point[0]
            code_lines = script_code.splitlines()[start:end+1]
            func_code = "\n".join(code_lines).strip()
            methods.append({
                "type": "function",
                "name": func_name,
                "content": func_code
            })
    return methods

def parse_code_file(file_path: str) -> list:
    file_ext = Path(file_path).suffix
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    if file_ext == ".vue":
        content = extract_script_block(content)

    chunks = get_method_chunks(content)
    for chunk in chunks:
        chunk["file_path"] = str(file_path)
    return chunks

def parse_directory(directory: str) -> list:
    all_chunks = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".vue") or file.endswith(".ts"):
                full_path = os.path.join(root, file)
                try:
                    chunks = parse_code_file(full_path)
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"Failed to parse {full_path}: {e}")
    return all_chunks

def save_as_jsonl(chunks: list, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            record = {
                "id": i,
                "file": chunk["file_path"],
                "name": chunk["name"],
                "type": chunk["type"],
                "code": chunk["content"]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    source_dir = "./sample_project"  # 분석할 루트 디렉토리
    output_file = "chunks.jsonl"

    chunks = parse_directory(source_dir)
    save_as_jsonl(chunks, output_file)
    print(f"✅ {len(chunks)} chunks saved to {output_file}")
