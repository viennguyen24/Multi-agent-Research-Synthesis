from langfuse import Langfuse
from langfuse.callback import CallbackHandler

class AgentLogger:
    """
    AgentLogger handles integration with Langfuse to provide observability 
    for the multi-agent graph and LLM responses natively.
    """
    def __init__(self):
        # Langfuse automatically grabs LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, 
        # and LANGFUSE_BASE_URL from environment variables.
        self.client = Langfuse()

    def get_langgraph_handler(self, **kwargs) -> CallbackHandler:
        """
        Returns a Langfuse CallbackHandler for LangGraph. 
        Hooks directly into the graph's config to trace node execution.
        """
        return CallbackHandler(**kwargs)

    def flush(self):
        """
        Flushes queued events to the Langfuse backend. 
        Should be called before the process exits.
        """
        self.client.flush()
