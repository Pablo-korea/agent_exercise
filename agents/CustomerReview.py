"""CustomerReview Agent: 고객 입장을 대변하는 Agent (기본 전문가)."""
from agents.Expert import agent_for, register_builtin

ROLE = "고객 전문가"
# 고객 입장 전문 Agent (공정성, 고객 영향, 민원·분쟁 가능성, 신뢰 관점)
PROMPT = (
    "너는 보험 가입 고객의 입장을 대변하는 고객 전문가야. "
    "공정성, 고객에게 미치는 영향, 민원·분쟁 가능성, 신뢰와 평판 관점에서만 검토해."
)

register_builtin("CustomerReview", "expert", ROLE, PROMPT, "🤝", "#34d399")
agent = agent_for({"name": "CustomerReview", "role": ROLE, "prompt": PROMPT})
