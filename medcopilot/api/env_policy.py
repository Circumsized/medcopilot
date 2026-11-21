"""环境分类(W1.3 生产默认拒绝)。

默认拒绝契约:``PROJECT_ENV`` 显式命中开发白名单才开放无鉴权读/写;
``production``/``prod`` 为识别到的生产;空值或未识别值(含错拼)一律
``unrecognized`` ⇒ 读写 fail-closed(即便 ADMIN_API_KEY 已配置也不放行——
错拼的环境必须失败而不是被猜)。纯函数,便于单测。
"""

from __future__ import annotations

DEV_ENVIRONMENTS = frozenset({"development", "dev", "test", "local"})
PROD_ENVIRONMENTS = frozenset({"production", "prod"})


def classify_env(raw: str | None) -> str:
    """返回 "dev" | "production" | "unrecognized"(大小写/空白不敏感)。"""
    value = (raw or "").strip().lower()
    if value in DEV_ENVIRONMENTS:
        return "dev"
    if value in PROD_ENVIRONMENTS:
        return "production"
    return "unrecognized"


def recognized_literals() -> list[str]:
    """/health 提示用:可设置的合法字面量(不含任何敏感信息)。"""
    return sorted(DEV_ENVIRONMENTS | PROD_ENVIRONMENTS)
