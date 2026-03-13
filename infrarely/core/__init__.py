"""infrarely.core — Core framework: Agent, Result, Config, Events, Decorators, Streaming."""

from infrarely.core.result import Result, Error, ErrorType
from infrarely.core.config import configure, get_config
from infrarely.core.decorators import tool, capability
from infrarely.core.agent import Agent, agent
