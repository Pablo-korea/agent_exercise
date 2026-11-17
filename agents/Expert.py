"""전문가 Agent 공통 팩토리 + Agent 목록(레지스트리).

- 기본 3명(LegalReview, CEO, CustomerReview)은 각자 파일에서 자기 정보를 register_builtin 으로 등록한다.
- 화면에서 추가한 전문가는 make_expert 로 만든다. (agents/{이름}.py 파일 없이 공통 팩토리 사용)
- 전문가는 항상 Full-Mesh: 목록에 있는 전문가는 설정 없이 서로 피드백을 주고받는다.
- 사건 진행 중에 Agent 를 고쳐도 그 사건은 '시작 시점 구성'을 유지한다 (begin_run 의 스냅샷)."""
import re
import threading

from deepagents import create_deep_agent
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from agents.common import (
    COMMON_RULES, MEMORY, ApiError, emit_event, get_model, note_excerpts, search_docs, unlink_agent,
)

MAX_EXPERTS = 6  # 전문가 최대 인원
MIN_EXPERTS = 2  # 전문가 최소 인원 (Full-Mesh 가 성립하려면 2명 이상)
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]{1,23}$")
COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")

# 약관 문서 사용 규칙: 모든 전문가 프롬프트 뒤에 붙는다
DOC_RULES = (
    "연결된 약관 문서가 있으면 search_terms Tool 로 관련 조항을 찾아 근거 조항(예: 제9조)을 짧게 인용해. "
    "연결된 문서가 없으면 이 Tool 을 쓰지 마."
)

_lock = threading.Lock()
SPECS: "dict[str, dict]" = {}  # 이름 → {name, kind, role, prompt, icon, color, builtin}


_seen_refs: dict[str, set] = {}  # 사건 ID → 이미 화면에 알린 (Agent, 문서, 조항)


# ===== 약관 검색 Tool (모든 Agent 가 가지고 있고, 자기에게 연결된 문서만 검색된다) =====
@tool
def search_terms(query: str, config: RunnableConfig) -> str:
    """연결된 약관 문서에서 질문과 관련된 조항을 검색한다. 조항 번호와 발췌를 돌려준다."""
    cfg = config["configurable"]
    agent, case_id = cfg.get("agent", ""), cfg.get("case_id", "")
    hits = search_docs(agent, query)  # 이 Agent 에 연결된 문서에서만 찾는다
    if not hits:
        return "연결된 약관 문서에서 관련 조항을 찾지 못했습니다."
    note_excerpts(case_id, agent, hits)  # 동료에게 피드백을 요청할 때 같이 보여 주기 위해 기록
    # 화면에 '📄 참조' 배지와 로그를 띄운다 (같은 조항은 사건 중 한 번만)
    seen = _seen_refs.setdefault(case_id, set())
    for h in hits:
        if (agent, h["doc"], h["clause"]) in seen:
            continue
        seen.add((agent, h["doc"], h["clause"]))
        emit_event(case_id, {
            "active": agent, "edges": [],
            "log": f"📄 {agent} 가 [{h['doc']}] {h['clause']} 참고",
            "refs": [{"agent": agent, "doc": h["doc"], "clause": h["clause"]}],
        })
    return "\n\n".join(f"[{h['doc']}] {h['clause']}\n{h['text']}" for h in hits)


# ===== Agent 만들기 =====
_cache: dict = {}  # (이름, 프롬프트) → Agent. 프롬프트가 바뀌면 새로 만든다


def make_expert(name: str, role: str, prompt: str):
    """전문가 Agent 하나를 만든다. 기본 3명도 화면에서 추가한 전문가도 모두 이 팩토리를 쓴다."""
    return create_deep_agent(
        model=get_model(),
        tools=[search_terms],
        checkpointer=MEMORY,  # 사건별 기억
        system_prompt=f"{prompt} {DOC_RULES} {COMMON_RULES}",
    )


def agent_for(spec: dict):
    """spec 에 맞는 Agent 를 돌려준다 (같은 이름·프롬프트면 다시 만들지 않는다)."""
    key = (spec["name"], spec["prompt"])
    if key not in _cache:
        _cache[key] = make_expert(spec["name"], spec["role"], spec["prompt"])
    return _cache[key]


# ===== 목록 관리 =====
def register_builtin(name: str, kind: str, role: str, prompt: str, icon: str, color: str) -> None:
    """기본 Agent 를 목록에 넣는다 (각 Agent 파일이 import 될 때 호출)."""
    SPECS[name] = {"name": name, "kind": kind, "role": role, "prompt": prompt,
                   "icon": icon, "color": color, "builtin": True}


def expert_count() -> int:
    return sum(1 for s in SPECS.values() if s["kind"] == "expert")


def list_agents() -> list[dict]:
    """Orchestrator 를 맨 앞에, 나머지는 등록 순서대로."""
    specs = sorted(SPECS.values(), key=lambda s: s["kind"] != "orchestrator")
    return [dict(s) for s in specs]


def agent_names() -> list[str]:
    return list(SPECS)


def _clean(body: dict, need_name: bool) -> dict:
    """요청 내용을 검사해서 정리한 값을 돌려준다. 잘못되면 ApiError(한국어 메시지)."""
    if not isinstance(body, dict):
        raise ApiError(400, "요청 형식이 올바르지 않습니다.")
    out = {}
    for key in (("name",) if need_name else ()) + ("role", "prompt", "icon", "color"):
        value = body.get(key)
        if not isinstance(value, str) or not value.strip():
            label = {"name": "이름", "role": "역할 이름", "prompt": "역할 지침", "icon": "아이콘", "color": "색상"}[key]
            raise ApiError(400, f"{label}을(를) 입력해 주세요.")
        out[key] = value.strip()
    if need_name:
        if out["name"].lower() == "user":
            raise ApiError(400, "'User' 는 예약된 이름이라 쓸 수 없습니다.")
        if not NAME_RE.match(out["name"]):
            raise ApiError(400, "이름은 영문으로 시작하는 영문·숫자 2~24자여야 합니다.")
    if len(out["role"]) > 30:
        raise ApiError(400, "역할 이름은 30자 이하여야 합니다.")
    if len(out["prompt"]) > 2000:
        raise ApiError(400, "역할 지침은 2000자 이하여야 합니다.")
    if len(out["icon"]) > 16:
        raise ApiError(400, "아이콘은 이모지 한 개로 입력해 주세요.")
    if not COLOR_RE.match(out["color"]):
        raise ApiError(400, "색상은 #rrggbb 형식이어야 합니다. 예: #6366f1")
    return out


def create_agent(body: dict) -> dict:
    data = _clean(body, need_name=True)
    with _lock:
        if data["name"].lower() in (n.lower() for n in SPECS):
            raise ApiError(409, "이미 사용 중인 이름입니다.")
        if expert_count() >= MAX_EXPERTS:
            raise ApiError(400, f"전문가는 최대 {MAX_EXPERTS}명까지 추가할 수 있습니다.")
        SPECS[data["name"]] = {**data, "kind": "expert", "builtin": False}
        return dict(SPECS[data["name"]])


def update_agent(name: str, body: dict) -> dict:
    data = _clean(body, need_name=False)
    with _lock:
        if name not in SPECS:
            raise ApiError(404, "Agent 를 찾을 수 없습니다.")
        SPECS[name].update(data)  # 이름은 바꿀 수 없다
        return dict(SPECS[name])


def delete_agent(name: str) -> None:
    with _lock:
        spec = SPECS.get(name)
        if spec is None:
            raise ApiError(404, "Agent 를 찾을 수 없습니다.")
        if spec["kind"] == "orchestrator":
            raise ApiError(400, "Orchestrator 는 삭제할 수 없습니다.")
        if expert_count() <= MIN_EXPERTS:
            raise ApiError(400, f"전문가는 최소 {MIN_EXPERTS}명이 필요해 삭제할 수 없습니다.")
        del SPECS[name]
    unlink_agent(name)  # 연결된 약관 문서에서도 즉시 뺀다


# ===== 사건별 스냅샷: 진행 중인 사건은 시작 시점의 구성을 유지한다 =====
_runs: dict[str, dict] = {}


def snapshot() -> dict:
    with _lock:
        specs = [dict(s) for s in SPECS.values()]
    return {
        "orchestrator": next(s for s in specs if s["kind"] == "orchestrator"),
        "experts": {s["name"]: s for s in specs if s["kind"] == "expert"},
    }


def begin_run(case_id: str) -> dict:
    _runs[case_id] = snapshot()
    return _runs[case_id]


def get_run(case_id: str) -> dict:
    return _runs.get(case_id) or snapshot()


def end_run(case_id: str) -> None:
    _runs.pop(case_id, None)
    _seen_refs.pop(case_id, None)
