"""CEO Agent: 보험회사 경영자 입장을 대변하는 Agent (기본 전문가)."""
from agents.Expert import agent_for, register_builtin

ROLE = "경영 전문가"
# 경영자 입장 전문 Agent (재무 영향, 손해율, 리스크, 선례 일관성, 회사 평판 관점)
PROMPT = (
    "너는 보험회사 경영자(CEO)의 입장을 대변하는 경영 전문가야. "
    "재무 영향, 손해율, 리스크, 선례 일관성, 회사 평판 관점에서만 검토해."
)

register_builtin("CEO", "expert", ROLE, PROMPT, "📊", "#fbbf24")
agent = agent_for({"name": "CEO", "role": ROLE, "prompt": PROMPT})
