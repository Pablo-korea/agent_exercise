"""FastAPI 서버: 화면(HTML)을 보여주고, 챗봇 요청을 Agent에게 전달한다."""
import json
import queue
import threading

from fastapi import Body, FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from agents import Expert
from agents.Orchestrator import get_orchestrator  # 사용자 프롬프트를 받는 Orchestrator Agent
from agents.common import (
    MAX_DOC_BYTES, DOCS, ApiError, add_document, delete_document, forget_case, get_document,
    public_doc, register_sink, thread_config, unregister_sink,
)

app = FastAPI()


# ===== 오류 응답: 항상 {"detail": "한국어 메시지"} =====
@app.exception_handler(ApiError)
async def api_error_handler(_req: Request, e: ApiError):
    return JSONResponse({"detail": e.detail}, status_code=e.status)


@app.exception_handler(RequestValidationError)
async def validation_handler(_req: Request, _e: RequestValidationError):
    return JSONResponse({"detail": "요청 형식이 올바르지 않습니다."}, status_code=422)


class ChatRequest(BaseModel):
    case_id: str  # 사건 ID: 페이지를 열 때마다 새로 만들어진다 (새로고침하면 바뀜 → 기억 초기화)
    message: str  # 이번에 입력한 메시지 (이전 대화는 서버 메모리의 Agent 기억에 있다)


@app.get("/")
def index():
    # frontend 폴더의 HTML 파일을 브라우저에 보여준다
    return FileResponse("frontend/index.html")


def line(**data) -> str:
    # 브라우저로 보낼 한 줄짜리 JSON (줄 단위로 읽기 쉽다)
    return json.dumps(data, ensure_ascii=False) + "\n"


def run_agent(case_id: str, message: str):
    # 어떤 Agent가 동작 중인지(active), 로그(log)를 한 줄씩 브라우저에 흘려보낸다
    yield line(type="status", active="Orchestrator", log="Orchestrator 시작", edges=[])
    snap = Expert.begin_run(case_id)  # 이 사건은 지금의 Agent 구성으로 끝까지 진행한다
    orchestrator = get_orchestrator(snap)
    config = thread_config(case_id, "Orchestrator")  # 기억 칸: 사건 ID + Orchestrator

    # Agent 는 별도 스레드에서 돌리고, 모든 진행 상황을 큐 하나로 모은다.
    # (Tool 안의 전문가들도 스레드로 동시에 일하므로, 큐를 써야 실시간으로 화면에 흘려보낼 수 있다)
    q: "queue.Queue" = queue.Queue()
    register_sink(case_id, q)

    def work():
        try:
            # stream_mode="updates": Agent 가 한 단계를 끝낼 때마다 결과를 받는다
            for chunk in orchestrator.stream({"messages": [{"role": "user", "content": message}]}, config, stream_mode="updates"):
                q.put(("updates", chunk))
        except Exception as e:  # 화면이 멈추지 않도록 오류도 화면으로 보낸다
            q.put(("error", f"{type(e).__name__}: {e}"))
        finally:
            q.put(("done", None))

    threading.Thread(target=work, daemon=True).start()

    reply = ""
    try:
        while True:
            kind, item = q.get()
            if kind == "done":
                break
            if kind == "error":
                yield line(type="status", active=None, log=f"오류: {item}", edges=[])
                reply = "처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
                continue
            if kind == "event":
                # 조율 과정 로그(의견 요약, 피드백, 조율 내용), edges(그래프의 선), refs(약관 참조)
                extra = {"refs": item["refs"]} if item.get("refs") else {}
                yield line(type="status", active=item["active"], log=item["log"], edges=item.get("edges", []), **extra)
                continue
            for node, update in item.items():
                msgs = (update or {}).get("messages")
                if not isinstance(msgs, list):
                    continue
                for m in msgs:
                    if node == "tools":
                        # Tool 실행이 끝났으니 다시 Orchestrator 차례
                        yield line(type="status", active="Orchestrator", log=f"{m.name} 완료 → Orchestrator 계속 진행", edges=[])
                    elif getattr(m, "tool_calls", None):
                        # Orchestrator 가 Tool 을 호출하려는 단계 (전문가가 동적이므로 인자에서 이름을 읽는다)
                        for call in m.tool_calls:
                            who = (call.get("args") or {}).get("expert") or "Orchestrator"
                            yield line(type="status", active=who, log=f"Orchestrator → Tool 호출: {call['name']} ({who})", edges=[])
                    elif getattr(m, "type", "") == "ai":
                        reply = m.text  # Tool 호출 없는 AI 메시지 = 최종 답변
        yield line(type="status", active=None, log="완료", edges=[])
        yield line(type="reply", text=reply)
    finally:
        unregister_sink(case_id)
        Expert.end_run(case_id)


@app.post("/chat")
def chat(req: ChatRequest):
    return StreamingResponse(run_agent(req.case_id, req.message), media_type="application/x-ndjson")


@app.post("/cases/{case_id}/end")
def end_case(case_id: str):
    # 페이지를 닫거나 새로고침하면 브라우저가 알려 준다 → 그 사건의 기억을 모두 지운다
    forget_case(case_id)
    return Response(status_code=204)


# ===== Agent 관리 API =====
@app.get("/api/agents")
def list_agents():
    return {"agents": Expert.list_agents()}


@app.post("/api/agents")
def create_agent(body: dict = Body(...)):
    return Expert.create_agent(body)


@app.put("/api/agents/{name}")
def update_agent(name: str, body: dict = Body(...)):
    return Expert.update_agent(name, body)


@app.delete("/api/agents/{name}")
def delete_agent(name: str):
    Expert.delete_agent(name)
    return Response(status_code=204)


# ===== 약관 문서 API (서버 메모리, 서버 재시작 전까지 유지) =====
def check_agents(names) -> list[str]:
    """연결할 Agent 이름 목록이 올바른지 검사한다."""
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ApiError(400, "연결할 Agent 는 이름 목록이어야 합니다.")
    unknown = [n for n in names if n not in Expert.SPECS]
    if unknown:
        raise ApiError(400, f"존재하지 않는 Agent 입니다: {', '.join(unknown)}")
    return list(dict.fromkeys(names))


@app.get("/api/documents")
def list_documents():
    return {"documents": [public_doc(d) for d in list(DOCS.values())]}


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...), agents: str = Form("[]")):
    try:
        names = json.loads(agents)
    except ValueError:
        raise ApiError(400, "연결할 Agent 형식이 올바르지 않습니다.")
    names = check_agents(names)
    data = await file.read(MAX_DOC_BYTES + 1)  # 5MB 를 넘는지만 알면 되므로 1바이트 더 읽는다
    return public_doc(add_document(file.filename or "", data, names))


@app.put("/api/documents/{doc_id}/agents")
def link_document(doc_id: str, body: dict = Body(...)):
    doc = get_document(doc_id)
    doc["agents"] = check_agents(body.get("agents"))  # 연결 Agent 전체 교체 → 이후 검색부터 바로 반영
    return public_doc(doc)


@app.delete("/api/documents/{doc_id}")
def remove_document(doc_id: str):
    delete_document(doc_id)
    return Response(status_code=204)
