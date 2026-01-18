import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI


def setup_environment():
    """
    Loads environment variables and checks for the required API key.
    """
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError(
            "GOOGLE_API_KEY not found. Please set it in your .env file."
        )


def main():
    """
    Initializes and runs the AI crew for content creation
    using the Gemini 2.5 Flash model.
    """
    setup_environment()

    # Gemini 2.5 Flash model
    llm = "gemini/gemini-2.5-flash"

    # Define Agents
    researcher = Agent(
        role="Senior Research Analyst",
        goal="Find and summarize the latest trends in AI.",
        backstory=(
            "You are an experienced research analyst with a knack "
            "for identifying key trends and synthesizing information."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    writer = Agent(
        role="Technical Content Writer",
        goal="Write a clear and engaging blog post based on research findings.",
        backstory=(
            "You are a skilled writer who can translate complex "
            "technical topics into accessible content."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    # Define Tasks
    research_task = Task(
        description=(
            "Research the top 3 emerging trends in Artificial Intelligence "
            "in 2024-2025. Focus on practical applications and potential impact."
        ),
        expected_output=(
            "A detailed summary of the top 3 AI trends, including "
            "key points and sources."
        ),
        agent=researcher,
    )

    writing_task = Task(
        description=(
            "Write a 500-word blog post based on the research findings. "
            "The post should be engaging and easy for a general audience to understand."
        ),
        expected_output=(
            "A complete 500-word blog post about the latest AI trends."
        ),
        agent=writer,
        context=[research_task],
    )

    # Create Crew
    blog_creation_crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        process=Process.sequential,
        llm=llm,
        verbose=True,
    )

    # Run Crew
    print("## Running the blog creation crew with Gemini 2.5 Flash... ##")

    try:
        result = blog_creation_crew.kickoff()
        print("\n------------------\n")
        print("## Crew Final Output ##")
        print(result)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
