# Migrating from CrewAI to InfraRely

## Why Switch?

| Feature | CrewAI | InfraRely |
|---------|--------|-----|
| Dependencies | LangChain + extras | 0 (stdlib only) |
| Agent definition | Role/Goal/Backstory | Name + tools |
| Task routing | Manual task assignment | Automatic intent classification |
| LLM bypass | Never | Automatic for factual queries |
| Error handling | Exceptions | Structured Results |
| Observability | Verbose/logging | Full traces, health, metrics |

---

## Side-by-Side: Multi-Agent Crew

### CrewAI
```python
from crewai import Agent, Task, Crew, Process

researcher = Agent(
    role="Senior Researcher",
    goal="Find comprehensive facts",
    backstory="You are an expert researcher...",
    verbose=True,
    llm=llm,
)
writer = Agent(
    role="Content Writer",
    goal="Write engaging articles",
    backstory="You are a skilled writer...",
    verbose=True,
    llm=llm,
)

research_task = Task(
    description="Research the topic of AI in education",
    expected_output="A list of key facts and statistics",
    agent=researcher,
)
write_task = Task(
    description="Write an article based on the research",
    expected_output="A well-written article",
    agent=writer,
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    verbose=True,
)
result = crew.kickoff()
```

### InfraRely
```python
import infrarely

researcher = infrarely.agent("researcher", description="Expert at finding facts")
writer = infrarely.agent("writer", description="Skilled content writer")

facts = researcher.run("Research AI in education")
article = writer.run("Write an article about AI in education", context=facts)
print(article.output)
```

---

## Side-by-Side: Parallel Execution

### CrewAI
```python
crew = Crew(
    agents=[agent1, agent2, agent3],
    tasks=[task1, task2, task3],
    process=Process.parallel,  # Not always reliable
)
result = crew.kickoff()
```

### InfraRely
```python
import infrarely

results = infrarely.parallel([
    (researcher, "Research topic A"),
    (analyst, "Analyze data B"),
    (writer, "Draft outline C"),
])

# Or async
results = await asyncio.gather(
    researcher.arun("Research topic A"),
    analyst.arun("Analyze data B"),
    writer.arun("Draft outline C"),
)
```

---

## Side-by-Side: Tools

### CrewAI
```python
from crewai_tools import SerperDevTool, WebsiteSearchTool

search_tool = SerperDevTool()
web_tool = WebsiteSearchTool()

agent = Agent(
    role="Researcher",
    tools=[search_tool, web_tool],
    ...
)
```

### InfraRely
```python
import infrarely

@infrarely.tool
def search(query: str) -> str:
    """Search the web."""
    return api.search(query)

@infrarely.tool
def scrape_website(url: str) -> str:
    """Scrape a website."""
    return api.scrape(url)

agent = infrarely.agent("researcher", tools=[search, scrape_website])
```

---

## Migration Checklist

- [ ] Replace `crewai` imports with `import infrarely`
- [ ] Replace `Agent(role=..., goal=..., backstory=...)` with `infrarely.agent("name")`
- [ ] Replace `Task(description=..., agent=...)` with `agent.run("description")`
- [ ] Replace `Crew(...).kickoff()` with sequential `agent.run()` calls or `infrarely.parallel()`
- [ ] Replace CrewAI tools with `@infrarely.tool` decorated functions
- [ ] Replace `Process.sequential` with chained `.run()` calls
- [ ] Replace `Process.parallel` with `infrarely.parallel()`
- [ ] Remove crewai from requirements
