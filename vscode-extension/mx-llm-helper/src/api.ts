import { SearchRequest, SearchResponse } from "./types";

export class ApiService {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  async search(
    question: string,
    selectedText?: string,
    currentFile?: SearchRequest["currentFile"]
  ): Promise<SearchResponse> {
    // Mock 응답 데이터
    const mockResult = `# 검색 결과

## 첫 번째 결과
이것은 첫 번째 검색 결과입니다.

## 두 번째 결과
이것은 두 번째 검색 결과입니다.

## 세 번째 결과
이것은 세 번째 검색 결과입니다.

${selectedText
        ? "> 선택된 코드 기반 검색"
        : currentFile
          ? "> 전체 파일 기반 검색"
          : "> 일반 검색"
      }`;

    // 실제 API 호출 대신 mock 데이터 반환
    return {
      question,
      status: "ok",
      result: mockResult,
    };


  }
}
