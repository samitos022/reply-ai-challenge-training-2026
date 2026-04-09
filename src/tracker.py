import os
import ulid
from langfuse import Langfuse
from dotenv import load_dotenv

# Carica le variabili d'ambiente
load_dotenv()

langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse")
)

def generate_session_id() -> str:
    """Genera un ID sessione univoco nel formato TEAM_NAME-ULID."""
    team_name = os.getenv('TEAM_NAME', 'myteam')
    return f"{team_name}-{ulid.new().str}"