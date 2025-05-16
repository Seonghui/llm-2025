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
  const apiService = new ApiService("http://localhost:8000");

  // 웹뷰 패널 생성
  let chatPanel: vscode.WebviewPanel | undefined;
  let currentSearchMode: "selected" | "file" | "general" = "general";
  let isPanelActive = false;

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
    if (!chatPanel || isPanelActive) {
      return;
    }

    const selectedText = getSelectedText();
    const currentFile = getCurrentEditorInfo();

    if (selectedText) {
      currentSearchMode = "selected";
    } else if (currentFile) {
      currentSearchMode = "file";
    } else {
      currentSearchMode = "general";
    }

    chatPanel.webview.postMessage({
      type: "updateSearchMode",
      mode: currentSearchMode,
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
      const htmlPath = path.join(context.extensionPath, "dist", "src", "chat.html");
      console.log("Loading HTML from:", htmlPath);
      const htmlContent = fs.readFileSync(htmlPath, "utf-8");
      chatPanel.webview.html = htmlContent;

      // 패널이 닫힐 때 처리
      chatPanel.onDidDispose(
        () => {
          console.log("Chat panel disposed");
          chatPanel = undefined;
          isPanelActive = false;
        },
        null,
        context.subscriptions
      );

      // 패널 활성화 상태 변경 이벤트
      chatPanel.onDidChangeViewState((e) => {
        isPanelActive = e.webviewPanel.active;
        if (!isPanelActive) {
          updateSearchMode();
        }
      });

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
                  currentSearchMode === "selected" ? selectedText : undefined,
                  currentSearchMode === "file" ? currentFile : undefined
                );
                console.log("API response:", response);

                // 웹뷰로 응답 전송
                if (chatPanel) {
                  if (response.status === "ok") {
                    chatPanel.webview.postMessage({
                      type: "addMessage",
                      text: response.result,
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
      const editorChangeDisposable = vscode.window.onDidChangeActiveTextEditor(() => {
        if (chatPanel && !isPanelActive) {
          updateSearchMode();
        }
      });

      const selectionChangeDisposable = vscode.window.onDidChangeTextEditorSelection(() => {
        if (chatPanel && !isPanelActive) {
          updateSearchMode();
        }
      });

      context.subscriptions.push(editorChangeDisposable, selectionChangeDisposable);
    }
  );

  context.subscriptions.push(openChatCommand);
  console.log("Extension activation completed");
}

// This method is called when your extension is deactivated
export function deactivate() {
  console.log("Extension is deactivating...");
}
