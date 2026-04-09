import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langfuse import get_client, observe, propagate_attributes
from langfuse.langchain import CallbackHandler

model = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="gpt-4o-mini",
    temperature=0.1,
    max_tokens=10
)

langfuse = get_client()

@observe()
def evaluate_citizen(session_id: str, citizen_id: str, citizen_data: str) -> int:
    handler = CallbackHandler()

    system_prompt = """
    You are 'The Eye', an advanced preventive AI in 2087.
    Analyze the citizen's longitudinal health and behavioral data.
    Identify subtle deviations, behavioral drifts, or suboptimal trajectories.
    
    RULES:
    - Output ONLY the number "1" if a personalized preventive support pathway is needed.
    - Output ONLY the number "0" if standard monitoring is sufficient.
    - Be analytical and objective. Do NOT output any other text.
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Citizen Data:\n{citizen_data}\n\nDecision (0 or 1):")
    ]

    try:
        with propagate_attributes(session_id=session_id):
            response = model.invoke(messages, config={"callbacks": [handler]})
        return 1 if "1" in response.content.strip() else 0
    except Exception as e:
        print(f"Errore analisi {citizen_id}: {e}")
        return 0