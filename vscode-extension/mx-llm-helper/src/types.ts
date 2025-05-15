export interface SearchResponse {
  question: string;
  status: "ok" | "error";
  result: string; // 마크다운 형식의 결과
}

export interface SearchRequest {
  question: string;
  selectedText?: string; // 선택된 텍스트
  currentFile?: {
    // 현재 파일 정보
    content: string;
    path: string;
    language: string;
  };
}
