import warnings
from typing import AsyncGenerator

# Suppress Pydantic protected namespace warnings (Google ADK internal)
warnings.filterwarnings("ignore", category=UserWarning)

from google.adk.agents import LlmAgent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event


# Custom non-LLM agent
class TaskExecutor(BaseAgent):
    """
    A specialized agent with custom, non-LLM behavior.
    """

    name: str = "TaskExecutor"
    description: str = "Executes a predefined task."

    async def _run_async_impl(
        self, context: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        Custom implementation logic for the task.
        """
        yield Event(
            author=self.name,
            content="Task finished successfully."
        )


# LLM-based sub-agent
greeter = LlmAgent(
    name="Greeter",
    model="gemini-2.5-flash",
    instruction="You are a friendly greeter."
)


# Custom logic sub-agent
task_doer = TaskExecutor()


# Parent / Supervisor agent
coordinator = LlmAgent(
    name="Coordinator",
    model="gemini-2.5-flash",
    description="A coordinator that can greet users and execute tasks.",
    instruction=(
        "When asked to greet, delegate to the Greeter. "
        "When asked to perform a task, delegate to the TaskExecutor."
    ),
    sub_agents=[
        greeter,
        task_doer
    ]
)


# Validate hierarchy
assert greeter.parent_agent == coordinator
assert task_doer.parent_agent == coordinator

print("Agent hierarchy created successfully with Gemini 2.5 Flash (warnings suppressed).")
