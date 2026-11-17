"""LegalReview Agent: 법률 관점에서 검토하는 Agent (기본 전문가)."""
from agents.Expert import agent_for, register_builtin

ROLE = "법률 전문가"
# 법률 검토 전문 Agent (약관, 법령, 판례 관점)
PROMPT = "너는 보험 분야 법률 검토 전문가야. 약관과 법령 관점에서만 검토해."

# 화면의 Agent 목록에 기본 전문가로 등록한다 (이후 화면에서 역할 지침을 고칠 수 있다)
register_builtin("LegalReview", "expert", ROLE, PROMPT, "⚖️", "#38bdf8")
agent = agent_for({"name": "LegalReview", "role": ROLE, "prompt": PROMPT})
