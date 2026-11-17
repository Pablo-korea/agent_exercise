"""Orchestrator Agent: 사용자의 프롬프트를 받아 대답하는 Agent.
전문가 Agent 는 expert_review Tool 로 호출한다.
Full-Mesh 구조: 어느 전문가가 호출되든, 그 전문가가 먼저 의견을 만들고
'나머지 모든 전문가'에게 피드백을 요청해 조율(최대 2번)한다.
전문가 목록은 화면에서 바뀔 수 있으므로, 사건을 시작할 때의 목록(스냅샷)을 쓴다."""
import re
from concurrent.futures import ThreadPoolExecutor

from deepagents import create_deep_agent
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

# 기본 전문가 3명을 목록에 등록하기 위해 import 한다 (import 될 때 register_builtin 이 실행됨).
# 화면 목록 순서(LegalReview, CEO, CustomerReview)가 곧 import 순서이므로 순서를 바꾸지 않는다.
from agents import LegalReview  # noqa: F401  isort: skip
from agents import CEO  # noqa: F401  isort: skip
from agents import CustomerReview  # noqa: F401  isort: skip
from agents.Expert import agent_for, get_run, register_builtin, search_terms
from agents.common import COMMON_RULES, MEMORY, ask, emit_event, get_model, pop_excerpts

MAX_ROUNDS = 2  # 조율은 최대 2번

ORCH_PROMPT = "너는 Orchestrator(심사팀장)야. 사용자의 질문에 한국어로 답해."
register_builtin("Orchestrator", "orchestrator", "심사팀장", ORCH_PROMPT, "🧑‍💼", "#a78bfa")


# ===== 로그 / 요약 도우미 =====
def make_logger(case_id: str):
    """화면 로그로 한 줄 보내는 함수를 만든다. active 는 지금 일하는 Agent 이름(들)."""

    def log(active, text, edges=None):
        # edges: 화면 그래프에 그릴 Agent 간 메시지 [{"from", "to", "kind", "label"}, ...]
        emit_event(case_id, {"active": active, "log": text, "edges": edges or []})

    return log


def pick(label: str, text: str) -> str:
    """답변에서 '라벨: 내용' 한 줄을 뽑는다. 없으면 첫 줄을 짧게 잘라 쓴다."""
    m = re.search(rf"{label}\s*:\s*(.+)", text)
    if m:
        return m.group(1).strip()
    first = next((l.strip() for l in text.splitlines() if l.strip()), "")
    return first[:100]


def agreed(text: str) -> bool:
    """피드백 첫 줄이 [동의] 면 동의로 본다."""
    first = next((l.strip() for l in text.splitlines() if l.strip()), "")
    return first.startswith("[동의]")


# ===== Full-Mesh 조율 =====
FEEDBACK_FORMAT = (
    "형식: 첫 줄은 [동의] 또는 [이견] 중 하나만. 둘째 줄은 '요약: ' 으로 시작하는 한 줄 요약. "
    "이어서 근거를 짧게. (이전에 네가 한 말과 달라진 점이 있으면 밝혀.)"
)


def coordinate(initiator: str, question: str, case_id: str) -> str:
    """발의자(initiator)가 먼저 의견을 만들고, 나머지 모든 전문가에게 피드백을 받아 조율한다.
    - 조율은 최대 MAX_ROUNDS 번, 모두 동의하면 바로 종료
    - 반환값은 Orchestrator 가 읽을 최종 의견 + 조율 경과"""
    log = make_logger(case_id)
    experts = get_run(case_id)["experts"]  # 사건 시작 시점의 전문가 목록
    init_spec = experts[initiator]
    init_agent = agent_for(init_spec)
    kind = f"{init_spec['role']} 의견"
    peers = {n: agent_for(s) for n, s in experts.items() if n != initiator}  # 나머지 모든 전문가
    peer_names = "·".join(peers)

    # 1단계: 발의자가 먼저 자기 의견을 만든다
    log(initiator, f"{initiator}: 초안 {kind} 작성 중")
    opinion = ask(
        init_agent, initiator, case_id,
        f"[사건]\n{question}\n\n네 관점의 {kind}을 작성해. 첫 줄은 '요약: ' 으로 시작하는 한 줄 요약, 이어서 근거.",
    )
    log(initiator, f"{initiator} 의견 요약: {pick('요약', opinion)}")
    evidence = pop_excerpts(case_id, initiator)  # 발의자가 실제로 조회한 약관 발췌 (없으면 빈 목록)

    history = []  # 조율 경과를 모아 두었다가 마지막에 Orchestrator 에게 돌려준다
    consensus = False
    for round_no in range(1, MAX_ROUNDS + 1):
        # 2단계: 발의자 → 나머지 모든 전문가에게 동시에 피드백을 요청한다 (Full-Mesh)
        log(
            list(peers), f"[조율 {round_no}/{MAX_ROUNDS}] {peer_names} 에게 피드백 요청",
            edges=[{"from": initiator, "to": n, "kind": "req", "label": "피드백 요청"} for n in peers],
        )
        # 약관에 연결되지 않은 전문가는 직접 검색할 수 없으니, 발의자가 조회한 원문 발췌를 같이 보여 준다
        evidence_text = (
            f"\n\n[{initiator} 이(가) 연결된 약관 문서에서 실제로 조회한 발췌 (원문)]\n" + "\n\n".join(evidence)
            if evidence else ""
        )
        prompt = (f"[사건]\n{question}\n\n[{initiator} 의견]\n{opinion}{evidence_text}\n\n"
                  f"이 의견에 대한 피드백을 줘. {FEEDBACK_FORMAT}")
        with ThreadPoolExecutor(max_workers=len(peers)) as pool:
            futures = {n: pool.submit(ask, a, n, case_id, prompt) for n, a in peers.items()}
            feedbacks = {n: f.result() for n, f in futures.items()}

        for name, fb in feedbacks.items():
            stance = "동의" if agreed(fb) else "이견"
            # 피드백은 발의자에게 돌아간다 (동의/이견 에 따라 화면 색이 달라진다)
            log(
                name, f"{name} 피드백 [{stance}]: {pick('요약', fb)}",
                edges=[{"from": name, "to": initiator, "kind": "agree" if stance == "동의" else "dissent", "label": stance}],
            )
            history.append(f"조율{round_no} {name} [{stance}]: {pick('요약', fb)}")

        # 모두 동의하면 조율을 끝낸다
        if all(agreed(fb) for fb in feedbacks.values()):
            consensus = True
            log("Orchestrator", f"[조율 {round_no}/{MAX_ROUNDS}] 모두 동의 → 조율 종료")
            break

        # 3단계: 발의자가 피드백을 반영해 의견을 수정한다
        log(initiator, f"[조율 {round_no}/{MAX_ROUNDS}] {initiator}: 피드백 반영해 의견 수정 중")
        fb_text = "\n\n".join(f"[{n} 피드백]\n{fb}" for n, fb in feedbacks.items())
        opinion = ask(
            init_agent, initiator, case_id,
            f"{fb_text}\n\n피드백을 반영해 {kind}을 수정해. 첫 줄은 '요약: ' 한 줄 요약, "
            "둘째 줄은 '조율: ' 으로 시작해 무엇을 받아들이고 무엇을 유지했는지 한 줄, 이어서 수정된 의견.",
        )
        evidence += [e for e in pop_excerpts(case_id, initiator) if e not in evidence]  # 수정하며 새로 조회한 조항
        log(initiator, f"조율된 의견 요약: {pick('요약', opinion)}")
        log(initiator, f"조율 내용: {pick('조율', opinion)}")
        history.append(f"조율{round_no} {initiator} 수정: {pick('조율', opinion)}")
    else:
        log("Orchestrator", f"최대 조율 {MAX_ROUNDS}번 도달 → 조율 종료 (일부 이견 남음)")

    result = "합의 도달" if consensus else f"최대 {MAX_ROUNDS}번 조율 후 종료 (이견 남음)"
    sources = ("\n\n[참고한 약관 발췌 (원문)]\n" + "\n\n".join(evidence)) if evidence else ""
    return f"[{initiator} 최종 의견]\n{opinion}{sources}\n\n[조율 결과] {result}\n" + "\n".join(history)


# ===== 전문가 호출 Tool: 전문가가 몇 명이든 이 Tool 하나로 부른다 =====
@tool
def expert_review(expert: str, question: str, config: RunnableConfig) -> str:
    """전문가 한 명에게 사건 검토를 맡길 때 호출한다. expert 는 전문가 이름(시스템 프롬프트의 목록 참고).
    그 전문가가 의견을 만들고 나머지 모든 전문가의 피드백을 받아 최대 2번 조율한 최종 의견과 조율 경과를 돌려준다."""
    case_id = config["configurable"]["case_id"]
    experts = get_run(case_id)["experts"]
    if expert not in experts:
        return f"'{expert}' 전문가는 없습니다. 선택 가능: {', '.join(experts)}"
    log = make_logger(case_id)
    # Orchestrator → 전문가 (화면 그래프의 선)
    log(expert, f"Orchestrator → {expert} 호출",
        edges=[{"from": "Orchestrator", "to": expert, "kind": "req", "label": "호출"}])
    result = coordinate(expert, question, case_id)
    # 전문가 → Orchestrator
    log("Orchestrator", f"{expert} → Orchestrator 결과 전달",
        edges=[{"from": expert, "to": "Orchestrator", "kind": "result", "label": "결과"}])
    return result


# ===== Orchestrator Agent =====
_cache: dict = {}


def get_orchestrator(snap: dict):
    """사건 스냅샷에 맞는 Orchestrator Agent. 전문가 구성이나 지침이 같으면 다시 만들지 않는다."""
    spec = snap["orchestrator"]
    roster = tuple((n, s["role"], s["prompt"]) for n, s in snap["experts"].items())
    key = (spec["prompt"], roster)
    if key not in _cache:
        lines = "\n".join(f"- {n}: {role}" for n, role, _ in roster)
        _cache[key] = create_deep_agent(
            model=get_model(),
            tools=[expert_review, search_terms],
            checkpointer=MEMORY,  # 사건별 기억 (대화 기록은 여기에 쌓인다)
            system_prompt=(
                f"{spec['prompt']} "
                f"전문가 목록:\n{lines}\n"
                "관점이 필요하면 expert_review Tool 에 전문가 이름과 사건 내용을 넣어 호출해. "
                "어느 전문가든 나머지 전문가들의 피드백까지 조율한 결과를 돌려주니, 그 결과를 바탕으로 최종 답을 정리해. "
                "한 가지 관점만 물으면 해당 전문가 한 명만 호출해. "
                "약관 근거는 전문가 결과에 붙은 '참고한 약관 발췌'를 우선 인용해. "
                "너에게 연결된 약관이 있을 때만 search_terms 가 결과를 주니, 결과가 비어 있어도 약관이 없다고 단정하지 마. "
                f"{COMMON_RULES}"
            ),
        )
    return _cache[key]
