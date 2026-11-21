"""急救红旗检测器(EmergencyDetector)v2 —— 确定性急救分诊的唯一权威源。

设计依据(W1.1):
- 日本 self-triage 安全审计(2026):red-flag 场景 100% hard under-triage,
  temperature=0 与 response-schema 均不能修复 ⇒ 急救门必须是确定性代码门;
- Nature Medicine(2026, ChatGPT Health 压测):真实急救 under-triage 51.6%,
  失败集中于"依赖病程演进的急症" ⇒ 增设 trajectory 红旗层;
- OSC red flags 研究(2025):商业自检工具仅覆盖 36.9% red flags(PCP 71.8%)
  ⇒ 词表覆盖度以权威红旗清单验收(scripts/emergency_eval.py 门禁)。

两项硬契约:
1. **归一化契约**:模式串与输入都先过 ``normalize_text``(NFKC → 去全部空白
   → casefold)再匹配。英文模式串必须按归一化后的无空格形态书写
   (``kill myself`` → ``killmyself``)。
2. **输入口径(anchoring 防护)**:急救可信输入只有当前用户 query;历史摘要、
   OCR、drug_context 等不可信输入永远不得触发或抑制急救判定——不可信输入
   不得驱动安全控制流。``evaluate`` 对非 ``user_query`` 的 source 抛
   ``ValueError``(fail-closed),不静默放行。

降级(保守偏召回):仅当三条同时成立才不触发——(a) 命中的规则全部
``history_demotable``;(b) 文本含叙事/病史标记;(c) 文本无任何当前急性标记。
信息疑问词(什么是/意思/介绍…)是独立的第三类标记,描述提问方式而非病史,
**不计入降级条件**:"我爸心梗了,心梗是什么意思"是当前急症的信息式追问。
自杀、药物过量、胸痛等急性家族永不降级。under-triage 是临床危害,
over-triage 仅损失一次 RAG 回答——fail toward hard trigger。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from medcopilot.common.logger import logger

_EMERGENCY_SOURCES = frozenset({"user_query"})


def normalize_text(text: str) -> str:
    """急救匹配前的归一化:全角→半角、去全部空白(含插入空白绕过)、英文大小写归一。"""
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = "".join(normalized.split())
    return normalized.casefold()


# 叙事/病史标记:出现这些词的句子更可能在叙述既往、影视、考题而非当前急症。
NARRATIVE_MARKERS = (
    "既往", "曾经", "以前", "病史", "家族史", "小时候", "多年前", "死于",
    "电视剧", "电影", "小说", "保险", "理赔", "考试", "题目", "科普", "报道",
)

# 信息疑问词:描述提问方式(求解释/求科普),不是病史叙事,不计入降级条件
# (当前急症的信息式追问必须触发)。与另两类标记不重叠,由测试钉死。
INFORMATIONAL_MARKERS = (
    "什么是", "介绍", "预防", "风险", "病因", "了解", "意思",
)

# 当前急性标记:出现即视为当前发作,拒绝降级。
ACTIVE_MARKERS = (
    "现在", "正在", "突发", "突然", "刚才", "刚刚", "持续", "越来越",
    "进行性", "加重", "更重", "伴", "出现", "疼", "痛", "不适", "麻", "晕",
    "呕", "喘", "憋", "动不了", "说不出话",
)


@dataclass(frozen=True)
class RedFlagRule:
    """一条急救红旗规则。patterns 作用于 normalize_text 之后的文本。"""

    rule_id: str
    family: str
    patterns: tuple[str, ...]
    history_demotable: bool = False


@dataclass(frozen=True)
class EmergencyVerdict:
    is_emergency: bool
    rule_ids: tuple[str, ...] = ()
    demoted: bool = False
    source: str = "user_query"
    families: tuple[str, ...] = field(default=())


def _rules() -> tuple[RedFlagRule, ...]:
    return (
        # ---------------- 心脑血管 ----------------
        RedFlagRule("cv.mi", "心脑血管", (
            "心肌梗死", "心肌梗塞", "心梗", "急性心肌梗死", "急性心梗",
        ), history_demotable=True),
        RedFlagRule("cv.acs.chestpain", "心脑血管", (
            "剧烈胸痛", "胸口剧痛", "胸痛难忍", "胸痛压榨", "压榨性胸痛",
            "胸口像压了块石头", "胸痛得打滚", "濒死感",
        )),
        RedFlagRule("cv.chestpain.bare", "心脑血管", (
            # 裸胸痛(胸痛是心源性急症最常见主诉,取严:出现即触发)。注意其模式
            # 自身含急性标记"疼/痛",叙事语境实际不可降级——保守方向的已知边界。
            "胸痛", "胸口疼痛", "胸口疼", "胸口痛", "胸部疼痛", "胸部疼", "胸部痛",
        ), history_demotable=True),
        RedFlagRule("cv.stroke", "心脑血管", (
            "中风", "脑卒中", "脑梗", "脑出血", "脑溢血", "脑血管意外", "急性脑血管病",
        ), history_demotable=True),
        RedFlagRule("cv.stroke.keywordfree", "心脑血管", (
            # FAST-120 式"无关键词中风描述":说不出话/脸歪/半边不动。
            "半边麻木", "半边身体麻木", "一侧麻木", "一侧肢体无力", "半边身体不能动",
            "半边身子不能动", "半边肢体不能动", "口齿不清", "说话不清", "说不出话",
            "嘴歪了", "嘴歪", "口角歪斜", "嘴眼歪斜", "舌头歪了",
            "半边.{0,6}(麻木|无力|不能动|动不了|没知觉)", "一侧.{0,6}(麻木|无力|不能动|动不了)",
        )),
        RedFlagRule("cv.thunderclap", "心脑血管", (
            # 雷击样头痛:剧烈头痛+喷射性呕吐(接受"头疼"口语写法,归一化已去空白)。
            "突发剧烈头痛.{0,6}喷射性呕吐", "剧烈头痛.{0,6}喷射性呕吐",
            "突发剧烈头疼.{0,6}喷射性呕吐", "剧烈头疼.{0,6}喷射性呕吐",
            "炸裂样头痛", "一生中最痛的头痛", "头痛得要炸开",
        )),
        RedFlagRule("cv.dissection", "心脑血管", (
            "主动脉夹层", "撕裂样疼痛.{0,10}(胸|背)", "(胸|背).{0,10}撕裂样疼痛",
        )),
        RedFlagRule("cv.hypertensive.crisis", "心脑血管", (
            "高血压危象", "血压骤升.{0,10}(昏|意识|不清|头痛|头疼)",
            # 症状联合而非数值诊断(AHA: ≥180/120 + 症状才 call 911):收缩压
            # ≥180,或 "130/125" 式舒张压 ≥120,且 8 字内出现症状;正常值不匹配。
            "血压.{0,6}(?:1[89]\\d|[2-9]\\d\\d).{0,8}(头晕|头痛|头疼|昏|意识|恶心|呕吐|胸闷|心悸|视物模糊|视力模糊)",
            "血压.{0,6}\\d{2,3}\\s*/\\s*1[2-9]\\d.{0,8}(头晕|头痛|头疼|昏|意识|恶心|呕吐|胸闷|心悸|视物模糊|视力模糊)",
        )),
        # ---------------- 神经系统 ----------------
        RedFlagRule("ns.seizure", "神经系统", (
            "抽搐", "惊厥", "癫痫持续", "抽过去了", "四肢抽动", "抽风",
        ), history_demotable=True),
        RedFlagRule("ns.febrile.seizure", "神经系统", (
            "高热惊厥", "高烧.{0,4}抽搐", "发烧.{0,4}抽搐", "高烧.{0,4}抽过去了",
        ), history_demotable=True),
        RedFlagRule("ns.conscious", "神经系统", (
            "意识不清", "昏迷", "晕厥不醒", "叫不醒", "呼之不应", "昏过去了",
            "不省人事", "失去意识",
        ), history_demotable=True),
        RedFlagRule("ns.meningitis", "神经系统", (
            "脖子硬.{0,6}(发热|发烧|高烧)", "颈项强直.{0,6}发热",
            "(发热|发烧).{0,6}脖子硬", "脖子发硬.{0,4}(发烧|发热)",
        ), history_demotable=True),
        # ---------------- 外伤/出血 ----------------
        RedFlagRule("tr.major.bleeding", "外伤", (
            "大出血", "流血不止", "止不住血", "喷血", "血往外喷", "出血凶猛",
            "(咳血|咯血|呕血|吐血|便血|尿血|流血).{0,6}(不止|不停|控制不住|好多|大量|很多)",
        ), history_demotable=True),
        RedFlagRule("tr.severe.trauma", "外伤", (
            "严重车祸", "车祸", "高坠", "坠楼", "高处坠落", "摔到头部", "摔到后脑",
            "头部重创", "头破血流", "开放性骨折", "骨头外露", "碾压伤",
        ), history_demotable=True),
        RedFlagRule("tr.burn", "外伤", (
            "大面积烧伤", "三度烧伤", "烧伤了.{0,8}(脸|手|呼吸道|喉咙)", "浑身着火",
        ), history_demotable=True),
        # ---------------- 中毒/服药 ----------------
        RedFlagRule("po.drug.overdose", "中毒服药", (
            # 多语序覆盖"服用过量安眠药""整瓶药吞下去了";安眠药单独出现不触发。
            "(过量|大量|整瓶|整盒|一把|一整瓶|一整板).{0,8}(安眠药|催眠药|安定|药|药物)",
            "(安眠药|催眠药|药物).{0,8}(过量|大量|整瓶|整盒|一把)",
            "吞了.{0,4}药", "误服.{0,4}(药|农药|老鼠药)", "服药过量", "药物过量",
            "吃过量", "超量服用",
        )),
        RedFlagRule("po.pesticide", "中毒服药", (
            "百草枯", "敌草快", "误服农药", "农药中毒", "喝农药", "杀虫剂中毒",
            "老鼠药", "服毒",
        ), history_demotable=True),
        RedFlagRule("po.general", "中毒服药", ("中毒",), history_demotable=True),
        RedFlagRule("po.other.toxin", "中毒服药", (
            "一氧化碳中毒", "煤气中毒", "煤烟中毒", "毒蛇咬伤", "毒蘑菇",
            "食物中毒.{0,6}(呕|昏|抽搐|休克)", "酒精中毒.{0,6}(昏迷|不清|叫不醒)",
        ), history_demotable=True),
        # ---------------- 精神(自伤):永不降级 ----------------
        RedFlagRule("mh.suicide", "精神(自伤)", (
            "自杀", "自残", "轻生", "割腕", "想死", "不想活了", "活不下去",
            "结束生命", "自我了断", "了结自己", "跳下去",
        )),
        RedFlagRule("mh.suicide.en", "精神(自伤)", (
            "suicide", "suicidal", "killmyself", "killme", "endmylife",
            "wanttodie", "wannadie", "selfharm", "harmmyself", "enditall",
        )),
        RedFlagRule("po.overdose.en", "中毒服药", (
            # 归一化后无空格形态;要求 bottle/wholebottle/pills 邻近动词,避免 "took two pills" 误伤。
            "overdose", "overdosed", "swallowed.{0,24}(bottle|pills)", "took.{0,12}wholebottle",
        )),
        # ---------------- 产科 ----------------
        RedFlagRule("ob.hemorrhage", "产科", (
            "怀孕.{0,10}(大出血|出血|流血|剧痛|肚子疼)", "孕期.{0,10}(大出血|出血|流血|剧痛)",
            "产前大出血", "阴道大出血", "阴道.{0,6}(出血|流血).{0,6}(不止|大量|很多|多)",
            "停经.{0,8}(剧痛|疼得打滚|流血)", "宫外孕", "胎动消失", "胎心没了",
            "破水.{0,4}(血|疼)",
        ), history_demotable=True),
        # ---------------- 呼吸 ----------------
        RedFlagRule("resp.arrest", "呼吸", (
            "窒息", "异物卡喉", "噎住了", "呼吸骤停", "喘不上来气", "喘不过气",
            "不能呼吸", "呼吸困难", "喉头水肿", "嘴唇发紫", "口唇发紫", "面色发紫", "口唇发绀",
        )),
        RedFlagRule("resp.status.asthma", "呼吸", (
            "哮喘持续状态", "喘得.{0,4}(说不了话|躺不下|睡不着)",
        )),
        # ---------------- 过敏/其他 ----------------
        RedFlagRule("al.shock", "其它", (
            "休克", "过敏性休克", "严重过敏.{0,6}(肿|喘|厥|紧)",
            "过敏.{0,6}(喉头水肿|休克)", "全身风团.{0,4}(喘|不过气)",
        ), history_demotable=True),
        RedFlagRule("other.dka", "其它", (
            "酮症酸中毒", "dka", "尿酮体.{0,6}(加|阳性|高)",
        ), history_demotable=True),
    )


# trajectory 层:Nature Medicine 指出的"病程演进型"急症(哮喘加重/DKA 的失败形态)。
_TRAJECTORY_PATTERNS = (
    "(逐渐|越来越|进行性|一天比一天|持续加重|不断加重|加重|恶化).{0,20}"
    "(喘|呼吸困难|喘不上气|胸痛|头痛|头疼|呕血|便血|腹痛|意识|麻木|无力|出血|肿胀|疼痛|说话不清)",
    "(喘|呼吸困难|胸痛|头痛|头疼|呕血|便血|腹痛|意识|麻木|无力|出血).{0,20}"
    "(逐渐|越来越|进行性|一天比一天|持续加重|不断加重|加重|恶化)",
    "(半小时|数小时|一天内|几分钟内).{0,20}(加重|恶化)",
)
_TRAJECTORY_RULE = RedFlagRule("traj.escalating-symptom", "病程演进", _TRAJECTORY_PATTERNS)


def compiled_patterns() -> tuple[re.Pattern[str], ...]:
    """全部规则的扁平化正则(模式串与顺序同 EmergencyDetector.RULES + TRAJECTORY_RULE)。

    跨规则重复的模式串按首次出现去重(策略工件要求无重复)。该属性被策略
    工件镜像(emergency_patterns)与 policy_check 等价守护引用。
    """
    patterns: list[str] = []
    for rule in (*_rules(), _TRAJECTORY_RULE):
        for pattern in rule.patterns:
            if pattern not in patterns:
                patterns.append(pattern)
    return tuple(re.compile(p) for p in patterns)


class EmergencyDetector:
    """确定性急救红旗检测。execution source = 本模块代码常量。"""

    RULES: tuple[RedFlagRule, ...] = _rules()
    TRAJECTORY_RULE: RedFlagRule = _TRAJECTORY_RULE

    @staticmethod
    def normalize_text(text: str) -> str:
        return normalize_text(text)

    @classmethod
    def evaluate(cls, query: str, *, source: str = "user_query") -> EmergencyVerdict:
        if source not in _EMERGENCY_SOURCES:
            raise ValueError(
                f"emergency triage input must be the current user query only, got source={source!r}"
            )
        text = normalize_text(query)
        if not text:
            return EmergencyVerdict(is_emergency=False, source=source)

        matched = [
            rule for rule in (*cls.RULES, cls.TRAJECTORY_RULE)
            if any(re.search(p, text) for p in rule.patterns)
        ]
        if not matched:
            return EmergencyVerdict(is_emergency=False, source=source)

        if cls._demote_if_narrative_only(matched, text):
            rule_ids = tuple(rule.rule_id for rule in matched)
            logger.warning(
                "Emergency pattern matched but demoted (narrative/history-only context): {}",
                ",".join(rule_ids),
            )
            return EmergencyVerdict(
                is_emergency=False, rule_ids=rule_ids, demoted=True, source=source,
                families=tuple(dict.fromkeys(rule.family for rule in matched)),
            )

        rule_ids = tuple(rule.rule_id for rule in matched)
        logger.warning(
            "Emergency red-flag rule fired: {}",
            ",".join(rule_ids),
        )
        return EmergencyVerdict(
            is_emergency=True, rule_ids=rule_ids, source=source,
            families=tuple(dict.fromkeys(rule.family for rule in matched)),
        )

    @classmethod
    def _demote_if_narrative_only(cls, matched: list[RedFlagRule], text: str) -> bool:
        """保守偏召回:三条同时成立才降级,缺一即触发。"""
        if any(not rule.history_demotable for rule in matched):
            return False
        if not any(marker in text for marker in NARRATIVE_MARKERS):
            return False
        return not any(marker in text for marker in ACTIVE_MARKERS)
