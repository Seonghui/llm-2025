// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
import * as vscode from "vscode";
import * as path from "path";
import * as fs from "fs";
import { ApiService } from "./api";
import { SearchRequest } from "./types";

// This method is called when your extension is activated
// Your extension is activated the very first time the command is executed
export function activate(context: vscode.ExtensionContext) {
  console.log("Extension activation started...");

  // API 서비스 초기화
  const apiService = new ApiService("http://localhost:3000"); // API 서버 URL을 적절히 수정하세요

  // 웹뷰 패널 생성
  let chatPanel: vscode.WebviewPanel | undefined;

  // 현재 에디터의 정보를 가져오는 함수
  function getCurrentEditorInfo(): SearchRequest["currentFile"] | undefined {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      return undefined;
    }

    return {
      content: editor.document.getText(),
      path: editor.document.fileName,
      language: editor.document.languageId,
    };
  }

  // 선택된 텍스트를 가져오는 함수
  function getSelectedText(): string | undefined {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      return undefined;
    }

    const selection = editor.selection;
    if (selection.isEmpty) {
      return undefined;
    }

    return editor.document.getText(selection);
  }

  // 검색 모드 업데이트 함수
  function updateSearchMode() {
    if (!chatPanel) {
      return;
    }

    const selectedText = getSelectedText();
    const currentFile = getCurrentEditorInfo();
    let mode: "selected" | "file" | "general" = "general";

    if (selectedText) {
      mode = "selected";
    } else if (currentFile) {
      mode = "file";
    }

    chatPanel.webview.postMessage({
      type: "updateSearchMode",
      mode,
    });
  }

  // 채팅 열기 명령어 등록
  let openChatCommand = vscode.commands.registerCommand(
    "mx-llm-helper.openChat",
    () => {
      console.log("Opening chat panel...");

      if (chatPanel) {
        console.log("Chat panel already exists, revealing...");
        chatPanel.reveal();
        return;
      }

      // 웹뷰 패널 생성
      console.log("Creating new chat panel...");
      chatPanel = vscode.window.createWebviewPanel(
        "mx-llm-helper.chatView",
        "LLM Chat",
        vscode.ViewColumn.Two,
        {
          enableScripts: true,
          retainContextWhenHidden: true,
        }
      );

      // HTML 내용 설정
      const htmlPath = path.join(context.extensionPath, "src", "chat.html");
      console.log("Loading HTML from:", htmlPath);
      const htmlContent = fs.readFileSync(htmlPath, "utf-8");
      chatPanel.webview.html = htmlContent;

      // 패널이 닫힐 때 처리
      chatPanel.onDidDispose(
        () => {
          console.log("Chat panel disposed");
          chatPanel = undefined;
        },
        null,
        context.subscriptions
      );

      // 웹뷰로부터 메시지 수신
      chatPanel.webview.onDidReceiveMessage(
        async (message) => {
          console.log("Received message from webview:", message);
          switch (message.type) {
            case "sendMessage":
              console.log("Processing message:", message.text);
              try {
                // 현재 에디터 정보와 선택된 텍스트 가져오기
                const selectedText = getSelectedText();
                const currentFile = getCurrentEditorInfo();

                // API 호출
                const response = await apiService.search(
                  message.text,
                  selectedText,
                  currentFile
                );
                console.log("API response:", response);

                // 웹뷰로 응답 전송
                if (chatPanel) {
                  if (response.status === "ok") {
                    // 결과를 문자열로 변환
                    const resultsText = response.results
                      .map((result) => JSON.stringify(result, null, 2))
                      .join("\n\n");

                    chatPanel.webview.postMessage({
                      type: "addMessage",
                      text: `검색 결과:\n\`\`\`\n${resultsText}\n\`\`\``,
                      isUser: false,
                    });
                  } else {
                    chatPanel.webview.postMessage({
                      type: "addMessage",
                      text: "죄송합니다. 검색 중 오류가 발생했습니다.",
                      isUser: false,
                    });
                  }
                }
              } catch (error) {
                console.error("Error processing message:", error);
                if (chatPanel) {
                  chatPanel.webview.postMessage({
                    type: "addMessage",
                    text: "죄송합니다. 요청을 처리하는 중 오류가 발생했습니다.",
                    isUser: false,
                  });
                }
              }
              break;
          }
        },
        undefined,
        context.subscriptions
      );

      // 초기 검색 모드 설정
      updateSearchMode();

      // 에디터 선택 변경 이벤트 구독
      context.subscriptions.push(
        vscode.window.onDidChangeActiveTextEditor(() => {
          updateSearchMode();
        }),
        vscode.window.onDidChangeTextEditorSelection(() => {
          updateSearchMode();
        })
      );
    }
  );

  context.subscriptions.push(openChatCommand);
  console.log("Extension activation completed");
}

// This method is called when your extension is deactivated
export function deactivate() {
  console.log("Extension is deactivating...");
}
