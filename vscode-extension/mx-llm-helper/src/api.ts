import { SearchRequest, SearchResponse } from "./types";
import axios from "axios";

export class ApiService {
  private baseUrl: string;
  private axiosInstance;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
    this.axiosInstance = axios.create({
      timeout: 300000, // 5분 타임아웃
      headers: {
        'Content-Type': 'application/json',
      }
    });
  }

  async search(
    question: string,
    selectedText?: string,
    currentFile?: SearchRequest["currentFile"],
    mode: '0' | '1' | '2' = '0'
  ): Promise<SearchResponse> {
    try {
      const response = await this.axiosInstance.post(`${this.baseUrl}/search`, {
        question,
        mode,
        selectedText,
        currentFile
      } as SearchRequest);

      return response.data as SearchResponse;
    } catch (error) {
      console.error("API call failed:", error);

      let errorMessage = "죄송합니다. API 호출 중 오류가 발생했습니다.";

      if (axios.isAxiosError(error)) {
        if (error.code === 'ETIMEDOUT') {
          errorMessage = "서버 응답 시간이 초과되었습니다. 서버가 실행 중인지 확인해주세요.";
        } else if (error.response) {
          errorMessage = `서버 오류: ${error.response.status} - ${error.response.statusText}`;
        } else if (error.request) {
          errorMessage = "서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.";
        }
      }

      return {
        question,
        status: "error",
        result: errorMessage,
      };
    }
  }
}
