"""infrarely.core — Core framework: Agent, Result, Config, Events, Decorators, Streaming."""

from infrarely.core.agent import Agent, agent
from infrarely.core.config import configure, get_config
from infrarely.core.decorators import capability, tool
from infrarely.core.result import Error, ErrorType, Result
