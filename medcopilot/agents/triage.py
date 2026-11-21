from __future__ import annotations

import re

from medcopilot.workflow.safety_guard import MedicalSafetyGuard


class TriageAgent:
    """Keyword-level intent router executed before any LLM/retrieval work.

    Routes: ``emergency`` -> hard red-line, ``non_medical`` -> polite refusal,
    ``medical`` -> RAG consultation.  The emergency check always wins so the
    safety guard cannot be bypassed by chitchat keywords.
    """

    EMERGENCY = "emergency"
    MEDICAL = "medical"
    NON_MEDICAL = "non_medical"

    NON_MEDICAL_KEYWORDS = (
        "天气", "股票", "彩票", "游戏", "编程", "python", "代码", "翻译",
        "足球", "篮球", "电影", "电视剧", "歌曲", "旅游攻略", "写一段", "作文", "写诗", "写小说",
    )
    MEDICAL_HINTS = (
        "生病", "就医", "医生", "医院", "吃药", "用药", "药品", "头痛", "胃痛", "胸痛", "腹痛",
        "发炎", "炎症", "发烧", "咳嗽", "感冒", "发热", "血压", "血糖", "剂量", "服用",
        "禁忌", "说明书", "症状", "手术", "过敏", "腹泻", "就诊", "门诊", "康复", "体检",
        "抢救", "梗死", "出血", "不良反应", "适应症",
    )

    @classmethod
    def classify(cls, query: str) -> str:
        if MedicalSafetyGuard.check_emergency(query)[0]:
            return cls.EMERGENCY
        text = query.lower()
        hit_non_medical = False
        for kw in cls.NON_MEDICAL_KEYWORDS:
            if kw.isascii() and kw.isalnum():
                if re.search(rf"\b{re.escape(kw)}\b", text):
                    hit_non_medical = True
                    break
            elif kw in text:
                hit_non_medical = True
                break
        hit_medical = any(hint in text for hint in cls.MEDICAL_HINTS)
        if hit_non_medical and not hit_medical:
            return cls.NON_MEDICAL
        return cls.MEDICAL
