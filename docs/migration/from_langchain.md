# Migrating from LangChain to InfraRely

## Why Switch?

| Feature | LangChain | InfraRely |
|---------|-----------|-----|
| Lines for basic agent | 50+ | 3 |
| Dependencies | 100+ packages | 0 (stdlib only) |
| LLM bypass for factual queries | No | Yes, automatic |
| Built-in observability | Plugin required | Default |
| Error handling | Exceptions | Structured Results |
| Memory | Manual setup | Built-in |

---

## Side-by-Side: Simple Agent

### LangChain (50 lines)
```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

@tool
def weather(city: str) -> str:
    """Get weather for a city."""
    return f"Sunny in {city}"

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    HumanMessage(content="{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_tools_agent(llm, [weather], prompt)
executor = AgentExecutor(agent=agent, tools=[weather], verbose=True)
result = executor.invoke({"input": "What's the weather in NYC?"})
print(result["output"])
```

### InfraRely (5 lines)
```python
import infrarely

@infrarely.tool
def weather(city: str) -> str:
    return f"Sunny in {city}"

agent = infrarely.agent("weather-bot", tools=[weather])
result = agent.run("What's the weather in NYC?")
print(result.output)  # "Sunny in NYC"
```

---

## Side-by-Side: RAG / Knowledge Base

### LangChain
```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

loader = TextLoader("./notes.txt")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500)
splits = splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever()
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
answer = qa.invoke("What is photosynthesis?")
```

### InfraRely
```python
import infrarely

agent = infrarely.agent("tutor")
agent.knowledge.add_documents("./notes.txt")
result = agent.run("What is photosynthesis?")
# LLM bypassed if knowledge confidence >= 85%
```

---

## Side-by-Side: Multi-Agent

### LangChain (CrewAI or custom)
```python
# Requires crewai or custom chain orchestration
from crewai import Agent, Task, Crew

researcher = Agent(role="Researcher", goal="Find facts", llm=llm)
writer = Agent(role="Writer", goal="Write articles", llm=llm)
task1 = Task(description="Research Mars", agent=researcher)
task2 = Task(description="Write article", agent=writer)
crew = Crew(agents=[researcher, writer], tasks=[task1, task2])
result = crew.kickoff()
```

### InfraRely
```python
import infrarely

researcher = infrarely.agent("researcher")
writer = infrarely.agent("writer")
facts = researcher.run("Find facts about Mars")
article = writer.run("Write article about Mars", context=facts)
```

---

## Side-by-Side: Error Handling

### LangChain
```python
try:
    result = executor.invoke({"input": "..."})
except Exception as e:
    # What kind of error? Good luck figuring out.
    print(f"Error: {e}")
```

### InfraRely
```python
result = agent.run("...")
if not result.success:
    print(result.error.type)       # ErrorType.TOOL_FAILURE
    print(result.error.message)    # "API timeout after 5s"
    print(result.error.suggestion) # "Increase timeout or add fallback"
```

---

## Migration Checklist

- [ ] Replace `langchain` imports with `import infrarely`
- [ ] Replace `@tool` with `@infrarely.tool`
- [ ] Replace `AgentExecutor` with `infrarely.agent()`
- [ ] Replace `executor.invoke()` with `agent.run()`
- [ ] Replace document loaders with `agent.knowledge.add_documents()`
- [ ] Replace vector stores with built-in knowledge (auto-managed)
- [ ] Replace `try/except` with `result.success` checks
- [ ] Remove prompt templates (InfraRely handles routing automatically)
- [ ] Remove `requirements.txt` entries for langchain packages
