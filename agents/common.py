"""모든 Agent가 함께 쓰는 공통 설정."""
import io
import queue
import re
import threading
import uuid
from collections import OrderedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import InMemorySaver

# .env 파일의 ANTHROPIC_API_KEY 를 읽어 온다 (Agent를 만들기 전에 한 번만 실행)
load_dotenv()

# 모든 Agent의 공통 시스템 프롬프트 규칙: 답변은 10줄 이하 + 기억/의견 수렴
COMMON_RULES = (
    "답변은 반드시 10줄을 넘기지 마. "
    "같은 사건에서 네가 이전에 한 말은 기억하고, 다른 전문가의 피드백을 반영해 의견을 조금씩 수렴해. "
    "같은 말을 반복하지 말고, 의견을 바꾸면 이유를, 유지하면 그 이유를 짧게 밝혀."
)


class ApiError(Exception):
    """화면에 한국어 메시지로 보여줄 오류. app.py 가 {"detail": 메시지} 와 상태 코드로 바꿔 응답한다."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


# ===== Agent 기억 (사건별) =====
# 서버 메모리에만 저장한다. 서버를 재시작하면 사라지고, DB/파일에는 남기지 않는다.
MEMORY = InMemorySaver()
MAX_CASES = 20  # 메모리가 무한히 커지지 않도록, 오래된 사건부터 지운다
_cases: "OrderedDict[str, None]" = OrderedDict()


def get_model() -> ChatAnthropic:
    """모델은 Sonnet 5.5 (effort=medium) 로 고정한다."""
    return ChatAnthropic(model="claude-sonnet-5-5", effort="medium")


def thread_config(case_id: str, name: str) -> dict:
    """'사건 ID + Agent 이름' 으로 기억 칸(thread)을 나눈다. 사건/Agent 가 다르면 기억이 섞이지 않는다.
    case_id / agent 는 Tool(search_terms 등)이 '누가 어느 사건에서 호출했는지' 알아내는 데 쓴다."""
    _cases[case_id] = None
    _cases.move_to_end(case_id)
    while len(_cases) > MAX_CASES:  # 너무 오래된 사건의 기억은 정리
        forget_case(next(iter(_cases)))
    return {"configurable": {"thread_id": f"{case_id}:{name}", "case_id": case_id, "agent": name}}


def forget_case(case_id: str) -> None:
    """한 사건의 모든 Agent 기억을 지운다 (새로고침/페이지 종료 시 호출).
    Agent 가 동적으로 늘고 줄어도 되도록, 이름 목록 대신 '사건 ID:' 로 시작하는 기억 칸을 모두 찾아 지운다."""
    _cases.pop(case_id, None)
    for key in [k for k in _excerpts if k[0] == case_id]:
        _excerpts.pop(key, None)
    for thread_id in [t for t in list(MEMORY.storage) if t.startswith(f"{case_id}:")]:
        MEMORY.delete_thread(thread_id)


def ask(agent, name: str, case_id: str, text: str) -> str:
    """Agent 를 invoke 로 실행한다. 같은 사건이면 이전 발언을 기억한 채로 답한다."""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": text}]},
        thread_config(case_id, name),
    )
    return result["messages"][-1].text


# ===== 화면으로 보내는 실시간 이벤트 =====
# Agent 들은 스레드 안에서 동시에 일하므로, 사건 ID 별 큐에 이벤트를 넣고
# app.py 가 그 큐에서 꺼내 브라우저로 흘려보낸다.
_sinks: dict[str, "queue.Queue"] = {}


def register_sink(case_id: str, q: "queue.Queue") -> None:
    _sinks[case_id] = q


def unregister_sink(case_id: str) -> None:
    _sinks.pop(case_id, None)


def emit_event(case_id: str, event: dict) -> None:
    """event = {"active", "log", "edges"?, "refs"?}. 화면이 듣고 있지 않으면 조용히 버린다."""
    q = _sinks.get(case_id)
    if q is not None:
        q.put(("event", event))


# ===== 약관 문서 (서버 메모리) =====
# 서버를 재시작하기 전까지는 유지한다. (새로고침/사건 종료로는 지워지지 않는다. 파일/DB 저장은 하지 않는다)
MAX_DOC_BYTES = 5 * 1024 * 1024  # 5MB
ALLOWED_EXT = (".txt", ".md", ".pdf", ".docx")
DOCS: "OrderedDict[str, dict]" = OrderedDict()
_docs_lock = threading.Lock()


def public_doc(doc: dict) -> dict:
    """화면으로 보낼 문서 정보 (본문 전체는 보내지 않는다)."""
    return {k: doc[k] for k in ("id", "name", "size", "agents", "preview")}


def extract_text(filename: str, data: bytes) -> str:
    """파일 종류별로 텍스트를 뽑는다."""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    try:
        if ext in (".txt", ".md"):
            for enc in ("utf-8-sig", "cp949"):
                try:
                    return data.decode(enc)
                except UnicodeDecodeError:
                    continue
            return data.decode("utf-8", errors="replace")
        if ext == ".pdf":
            from pypdf import PdfReader
            return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
        if ext == ".docx":
            import docx
            d = docx.Document(io.BytesIO(data))
            lines = [p.text for p in d.paragraphs]
            for t in d.tables:  # 표 안의 글자도 포함
                lines += [c.text for row in t.rows for c in row.cells]
            return "\n".join(lines)
    except Exception:
        raise ApiError(400, f"'{filename}' 파일을 읽을 수 없습니다. 손상된 파일은 아닌지 확인해 주세요.")
    raise ApiError(400, "지원하지 않는 형식입니다. .txt .md .pdf .docx 파일만 올릴 수 있습니다.")


def add_document(filename: str, data: bytes, agents: list[str]) -> dict:
    name = (filename or "").replace("\\", "/").rsplit("/", 1)[-1] or "문서"
    if not name.lower().endswith(ALLOWED_EXT):
        raise ApiError(400, "지원하지 않는 형식입니다. .txt .md .pdf .docx 파일만 올릴 수 있습니다.")
    if len(data) > MAX_DOC_BYTES:
        raise ApiError(413, "파일이 너무 큽니다. 5MB 이하만 올릴 수 있습니다.")
    text = extract_text(name, data).strip()
    if not text:
        raise ApiError(400, "문서에서 글자를 읽지 못했습니다. (스캔한 이미지 PDF는 지원하지 않습니다)")
    doc = {
        "id": uuid.uuid4().hex[:8], "name": name, "size": len(data), "agents": list(dict.fromkeys(agents)),
        "preview": re.sub(r"\s+", " ", text)[:160], "text": text,
    }
    with _docs_lock:
        DOCS[doc["id"]] = doc
    return doc


def get_document(doc_id: str) -> dict:
    doc = DOCS.get(doc_id)
    if doc is None:
        raise ApiError(404, "문서를 찾을 수 없습니다.")
    return doc


def delete_document(doc_id: str) -> None:
    get_document(doc_id)
    with _docs_lock:
        DOCS.pop(doc_id, None)


def unlink_agent(name: str) -> None:
    """Agent 가 삭제되면 모든 문서의 연결에서 뺀다."""
    with _docs_lock:
        for doc in DOCS.values():
            doc["agents"] = [a for a in doc["agents"] if a != name]


# ----- 약관 검색: 연결된 문서에서만, 조항 단위로 -----
# '제9조(보험금의 지급)' 처럼 줄 앞에 나오는 조항 제목을 기준으로 문서를 나눈다
CLAUSE_RE = re.compile(r"^[ \t]*(제\s*\d+\s*조(?:의\s*\d+)?)[ \t]*(?:[\(（]([^\)）\n]{0,30})[\)）])?", re.M)


def split_clauses(text: str) -> list[tuple[str, str]]:
    """[(조항 이름, 조항 본문), ...]. 조항 제목이 없으면 문단 단위로 나눈다."""
    ms = list(CLAUSE_RE.finditer(text))
    if not ms:
        return [("", p.strip()) for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(text)
        label = re.sub(r"\s+", "", m.group(1)) + (f"({m.group(2).strip()})" if m.group(2) else "")
        chunks.append((label, text[m.start():end].strip()))
    return chunks


def score(query: str, chunk: str) -> int:
    """질문 단어가 조항에 많이 들어 있을수록 높은 점수. 한국어는 단어가 붙어 있어 두 글자씩도 비교한다."""
    s = 0
    for tok in re.findall(r"\w+", query):
        if len(tok) < 2:
            continue
        if tok in chunk:
            s += 3
        else:
            s += sum(1 for i in range(len(tok) - 1) if tok[i:i + 2] in chunk)
    return s


def search_docs(agent: str, query: str, limit: int = 3) -> list[dict]:
    """agent 에 연결된 문서에서만 관련 조항을 찾는다. [{doc, clause, text}, ...]"""
    hits = []
    for doc in list(DOCS.values()):
        if agent not in doc["agents"]:
            continue
        for label, body in split_clauses(doc["text"]):
            sc = score(query, body)
            if sc > 0:
                hits.append((sc, doc["name"], label, body))
    hits = [h for h in hits if h[0] >= MIN_SCORE]  # 두 글자가 우연히 겹친 정도의 조항은 제외
    hits.sort(key=lambda h: -h[0])
    return [{"doc": d, "clause": c or "(조항 번호 없음)", "text": b[:300]} for _, d, c, b in hits[:limit]]


MIN_SCORE = 3  # 단어가 통째로 한 번 맞는 정도의 점수

# 발의자가 실제로 조회한 약관 발췌. 문서에 연결되지 않은 동료 Agent 는 직접 검색할 수 없으므로,
# 피드백을 요청할 때 이 발췌를 같이 보여 줘서 '검증할 수 없다'는 엉뚱한 반박을 막는다.
_excerpts: dict[tuple[str, str], list[str]] = {}


def note_excerpts(case_id: str, agent: str, hits: list[dict]) -> None:
    bucket = _excerpts.setdefault((case_id, agent), [])
    for h in hits:
        item = f"[{h['doc']}] {h['clause']}\n{h['text']}"
        if item not in bucket:
            bucket.append(item)


def pop_excerpts(case_id: str, agent: str) -> list[str]:
    return _excerpts.pop((case_id, agent), [])
