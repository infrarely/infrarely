import os


def test_basic_agent_run():
    import infrarely

    infrarely.configure(llm_provider="groq", api_key=os.getenv("GROQ_API_KEY"))
    agent = infrarely.agent("test")
    result = agent.run("What is 2+2?")
    assert result is not None
    assert result.output is not None
    assert result.error is None or result.success is True
