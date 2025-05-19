export interface SearchResponse {
  question: string;
  status: "ok" | "error";
  result: string; // 마크다운 형식의 결과
}

export interface SearchRequest {
  question: string;
  mode: '0' | '1' | '2';  // '0': general, '1': file, '2': selected
  selectedText?: string;
  currentFile?: {
    // 현재 파일 정보
    content: string;
    path: string;
    language: string;
  };
}
