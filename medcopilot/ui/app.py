"""Gradio front-end for Med-Copilot.

Slice 1 decoupled the UI from the clinical workflow: this module is a thin
consumer of the FastAPI service. It owns no chunker, retriever, OCR or agent
objects — only input widgets, the chat display and the opaque ``session_id``
returned by the API. All errors are rendered as stable, non-sensitive messages.
"""

from __future__ import annotations

import gradio as gr

from medcopilot.common.config import config
from medcopilot.ui.api_client import APIClientError, MedicalAPIClient
from medcopilot.workflow.safety_guard import DISCLAIMER_BANNER


def _build_client() -> MedicalAPIClient:
    """One client per app process; endpoint/timeout/Bearer come from AppConfig."""
    return MedicalAPIClient(
        base_url=config.med_api_base_url,
        api_key=config.admin_api_key,
        timeout_seconds=config.med_api_timeout_seconds,
    )


def handle_consultation(
    client: MedicalAPIClient,
    user_message: str | None,
    image_path: str | None,
    chat_history: list[dict[str, str]] | None,
    session_id: str | None,
) -> tuple[str, None, list[dict[str, str]], str | None]:
    """Call the API and fold the result into the chat history.

    Returns ``(msg, image, history, session_id)`` for Gradio output wiring.
    Any client error renders as a standalone assistant message; nothing partial
    is appended, and ``session_expired`` clears the local opaque id so the next
    consultation starts a fresh server-side session.
    """
    chat_history = list(chat_history or [])
    message = (user_message or "").strip()
    if not message:
        return "", None, chat_history, session_id

    try:
        if image_path:
            result = client.consult_with_image(message, image_path, session_id=session_id)
        else:
            result = client.consult_text(message, session_id=session_id)
    except APIClientError as exc:
        chat_history.append({"role": "assistant", "content": str(exc)})
        cleared = None if exc.code == "session_expired" else session_id
        return "", None, chat_history, cleared

    answer = result["answer"]
    if result.get("sources"):
        sections = ", ".join(str(s.get("section", "资料")) for s in result["sources"])
        answer += f"\n\n📚 **参考医学依据**: {sections}"
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": answer})
    return "", None, chat_history, result["session_id"]


def clear_conversation() -> tuple[str, None, list, None]:
    # Clearing the visible chat also drops the opaque session id, so the next
    # message cannot silently continue the previous conversation.
    return "", None, [], None


def build_app():
    client = _build_client()

    def respond(user_message, image_path, chat_history, session_id):
        return handle_consultation(client, user_message, image_path, chat_history, session_id)

    with gr.Blocks(title="妙手回春 - 智能临床医学用药咨询系统") as demo:
        session_state = gr.State(value=None)
        gr.Markdown(
            "# 🏥 妙手回春 (Med-Copilot) 临床辅助与智能用药问答系统\n"
            "支持 **药盒拍照 OCR 识别**、**智能分诊路由**、**BM25+Dense 混合检索 (RRF)** 与 **三重安全护栏**。"
        )
        # 常驻免责横幅(W5.7):在用户提问之前就展示,而不是只在回答末尾追加。
        # 文本来自 safety_guard.DISCLAIMER_BANNER(与 API 追加的免责同源)。
        gr.Markdown(DISCLAIMER_BANNER)
        chatbot = gr.Chatbot(label="医患对话窗口", height=500)
        with gr.Row():
            image = gr.Image(type="filepath", label="上传药盒照片（可选）", scale=4)
            msg = gr.Textbox(
                placeholder="请输入您的症状或用药疑问（如：布洛芬胃痛能吃吗？心肌梗死如何急救？）",
                scale=8,
            )
        with gr.Row():
            submit_btn = gr.Button("提交咨询", variant="primary")
            clear_btn = gr.Button("清空记录")

        msg.submit(respond, [msg, image, chatbot, session_state], [msg, image, chatbot, session_state])
        submit_btn.click(respond, [msg, image, chatbot, session_state], [msg, image, chatbot, session_state])
        clear_btn.click(clear_conversation, None, [msg, image, chatbot, session_state], queue=False)

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
