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
  vscode.window.showInformationMessage("Extension activation started...");

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
    if (!chatPanel) {
      return;
    }

    const selectedText = getSelectedText();
    const currentFile = getCurrentEditorInfo();

    let fileName: string | undefined = undefined;
    let lineInfo: string | undefined = undefined;

    if (selectedText) {
      currentSearchMode = "selected";
      if (currentFile) {
        fileName = path.basename(currentFile.path);
        const editor = vscode.window.activeTextEditor;
        if (editor) {
          const start = editor.selection.start.line + 1;
          const end = editor.selection.end.line + 1;
          lineInfo = start === end ? `${start}줄` : `${start}-${end}줄`;
        }
      }
    } else if (currentFile) {
      currentSearchMode = "file";
      fileName = path.basename(currentFile.path);
    } else {
      currentSearchMode = "general";
    }

    console.log("Updating search mode with data:", {
      mode: currentSearchMode,
      fileName,
      lineInfo,
      selectedText,
      currentFile
    });

    chatPanel.webview.postMessage({
      type: "updateSearchMode",
      mode: currentSearchMode,
      fileName,
      lineInfo,
      selectedText,
      currentFile
    });
  }

  // 채팅 열기 명령어 등록
  let openChatCommand = vscode.commands.registerCommand(
    "mx-llm-helper.openChat",
    () => {
      vscode.window.showInformationMessage("Opening chat panel...");

      if (chatPanel) {
        vscode.window.showInformationMessage(
          "Chat panel already exists, revealing..."
        );
        chatPanel.reveal();
        return;
      }

      // 웹뷰 패널 생성
      vscode.window.showInformationMessage("Creating new chat panel...");
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
      const htmlPath = path.join(
        context.extensionPath,
        "dist",
        "src",
        "chat.html"
      );
      vscode.window.showInformationMessage("Loading HTML from:", htmlPath);
      const htmlContent = fs.readFileSync(htmlPath, "utf-8");
      chatPanel.webview.html = htmlContent;

      // 패널이 닫힐 때 처리
      chatPanel.onDidDispose(
        () => {
          vscode.window.showInformationMessage("Chat panel disposed");
          chatPanel = undefined;
          isPanelActive = false;
        },
        null,
        context.subscriptions
      );

      // 패널 활성화 상태 변경 이벤트
      chatPanel.onDidChangeViewState((e) => {
        isPanelActive = e.webviewPanel.active;
        updateSearchMode(); // 활성화 상태와 관계없이 모드 업데이트
      });

      // 웹뷰로부터 메시지 수신
      chatPanel.webview.onDidReceiveMessage(
        async (message) => {
          console.log("Received message from webview:", message);
          switch (message.type) {
            case "sendMessage":
              console.log("Processing message with data:", {
                text: message.text,
                selectedText: message.selectedText,
                currentFile: message.currentFile,
                currentSearchMode
              });
              try {
                // WebView에서 전달받은 selectedText와 currentFile 정보 사용
                const selectedText = message.selectedText || getSelectedText();
                const currentFile = message.currentFile || getCurrentEditorInfo();

                console.log("Using data for API call:", {
                  selectedText,
                  currentFile,
                  currentSearchMode
                });

                // API 호출
                const response = await apiService.search(
                  message.text,
                  currentSearchMode === "selected" ? selectedText : undefined,
                  (currentSearchMode === "file" || currentSearchMode === "selected") ?
                    currentFile ? {
                      ...currentFile,
                      path: path.basename(currentFile.path)
                    } : undefined
                    : undefined,
                  currentSearchMode === "general"
                    ? "0"
                    : currentSearchMode === "file"
                      ? "1"
                      : "2"
                );

                console.log("Search request data:", {
                  question: message.text,
                  mode: currentSearchMode === "general" ? "0" : currentSearchMode === "file" ? "1" : "2",
                  selectedText: currentSearchMode === "selected" ? selectedText : undefined,
                  currentFile: (currentSearchMode === "file" || currentSearchMode === "selected") ?
                    currentFile ? {
                      ...currentFile,
                      path: path.basename(currentFile.path)
                    } : undefined
                    : undefined
                });

                vscode.window.showInformationMessage(
                  "API response: " + response.result
                );

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
            case "inputFocus": // 입력창 포커스 이벤트 처리
              // 현재 에디터 정보 가져오기
              const editor = vscode.window.activeTextEditor;
              if (editor) {
                const selectedText = getSelectedText();
                const currentFile = getCurrentEditorInfo();

                // 검색 모드 업데이트
                if (selectedText) {
                  currentSearchMode = "selected";
                } else if (currentFile) {
                  currentSearchMode = "file";
                } else {
                  currentSearchMode = "general";
                }

                // WebView로 정보 전송
                chatPanel?.webview.postMessage({
                  type: "updateSearchMode",
                  mode: currentSearchMode,
                  fileName: currentFile ? path.basename(currentFile.path) : undefined,
                  lineInfo: selectedText ? `${editor.selection.start.line + 1}줄` : undefined,
                  selectedText,
                  currentFile
                });
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
      const editorChangeDisposable = vscode.window.onDidChangeActiveTextEditor(
        () => {
          if (chatPanel && !isPanelActive) {
            updateSearchMode();
          }
        }
      );

      const selectionChangeDisposable =
        vscode.window.onDidChangeTextEditorSelection(() => {
          if (chatPanel && !isPanelActive) {
            updateSearchMode();
          }
        });

      context.subscriptions.push(
        editorChangeDisposable,
        selectionChangeDisposable
      );
    }
  );

  context.subscriptions.push(openChatCommand);
  vscode.window.showInformationMessage("Extension activation completed");
}

// This method is called when your extension is deactivated
export function deactivate() {
  vscode.window.showInformationMessage("Extension is deactivating...");
}
