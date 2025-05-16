# LLM 뺀 버전
import os
import re
import glob
import json

# ✅ Vue 파일 청크 추출 함수
def extract_chunks_from_vue(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # <template> 추출
    template_match = re.search(r"<template>(.*?)</template>", content, re.DOTALL)
    template = template_match.group(1).strip() if template_match else ""

    # <script> 추출
    script_match = re.search(r"<script[^>]*>(.*?)</script>", content, re.DOTALL)
    script = script_match.group(1).strip() if script_match else ""

    # 클래스명 추출
    class_match = re.search(r"export default class (\w+)", script)
    class_name = class_match.group(1) if class_match else "Unknown"

    # ✅ methods 코드 블록 추출 (이름만 추출)
    method_names = re.findall(
        r"(?:@\w+\(.*?\)\s*)*(?:public\s+|private\s+|protected\s+)?(?:async\s+)?(?:get|set)?\s*(\w+)\s*\([^)]*\)\s*(?::\s*[^\{]+)?\{",
        script,
        re.MULTILINE
    )

    # props & emits 그대로 유지
    props = re.findall(r"@Prop\(.*?\)\s+(\w+)!:.*?;", script)
    emits = re.findall(r'this\.\$emit\(["\'](.*?)["\']', script)

    # ✅ 임베딩용 텍스트 구성 (요약 제거)
    summary = f"""
Vue 컴포넌트 이름: {class_name}
파일 경로: {file_path}

[Template HTML]
{template}

[Script Code]
{script}

[Props]: {', '.join(props) if props else '없음'}
[Methods]: {', '.join(method_names) if method_names else '없음'}
[Emits]: {', '.join(emits) if emits else '없음'}
""".strip()

    return {
        "filePath": file_path,
        "template": template,
        "script": script,
        "className": class_name,
        "methods": method_names,
        "properties": list(set(props)),
        "emits": list(set(emits)),
        "embeddingText": summary
    }

# ✅ 전체 디렉토리 대상 추출
def extract_all_vue_chunks(root_dir):
    vue_files = glob.glob(os.path.join(root_dir, "**/*.vue"), recursive=True)
    chunks = []
    for file_path in vue_files:
        try:
            chunk = extract_chunks_from_vue(file_path)
            chunks.append(chunk)
        except Exception as e:
            print(f"⚠️ {file_path} 파싱 실패: {e}")
    return chunks

# ✅ 사용 예시
if __name__ == "__main__":
    repo_dir = "../repo/vue-ts-realworld-app"
    output_file = "../data/vue_chunks.json"

    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    result = extract_all_vue_chunks(repo_dir)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ 총 {len(result)}개 청크 추출 완료 → {output_file}")