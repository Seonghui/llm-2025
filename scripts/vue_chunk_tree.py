# pip install tree_sitter
# pip install tree_sitter_languages

import re
from tree_sitter_languages import get_language, get_parser

# TypeScript용 파서 준비
language = get_language('typescript')
parser = get_parser('typescript')

def extract_script_block(vue_code: str) -> str:
    match = re.search(r"<script[^>]*>(.*?)</script>", vue_code, re.DOTALL)
    return match.group(1).strip() if match else ""

def get_method_chunks(script_code: str) -> list:
    parser.set_language(language)
    tree = parser.parse(bytes(script_code, "utf8"))
    root = tree.root_node

    methods = []
    for class_decl in root.walk():
        if class_decl.type == "class_declaration":
            for child in class_decl.named_children:
                if child.type == "method_definition":
                    start = child.start_point[0]
                    end = child.end_point[0]
                    lines = script_code.splitlines()[start:end+1]
                    code_chunk = "\n".join(lines).strip()
                    methods.append(code_chunk)
    return methods

# 🔧 테스트
if __name__ == "__main__":
    with open("ArticleFavoritesButton.vue", encoding="utf-8") as f:
        vue_code = f.read()

    script = extract_script_block(vue_code)
    method_chunks = get_method_chunks(script)

    for i, chunk in enumerate(method_chunks, 1):
        print(f"\n--- Method {i} ---\n{chunk}")
