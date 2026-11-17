"""Deep Agents를 사용한 가장 간단한 App."""
import os

from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain_anthropic import ChatAnthropic

# .env 의 ANTHROPIC_API_KEY 를 항상 참고한다.
load_dotenv()
if not os.getenv("ANTHROPIC_API_KEY") or "여기에" in os.environ["ANTHROPIC_API_KEY"]:
    raise SystemExit(".env 파일에 ANTHROPIC_API_KEY 를 입력해 주세요.")

# 모델 고정: Sonnet 5.5, Effort: Medium
MODEL = ChatAnthropic(model="claude-sonnet-5-5", effort="medium")

# 프롬프트는 소스 코드에 직접 작성
PROMPT = "보험금 지급 여부를 심사할 때 고려해야 할 핵심 요소 3가지를 간단히 알려줘."

agent = create_deep_agent(
    model=MODEL,
    system_prompt="너는 보험심사 업무를 돕는 한국어 어시스턴트야.",
)

result = agent.invoke({"messages": [{"role": "user", "content": PROMPT}]})
print(result["messages"][-1].content)
