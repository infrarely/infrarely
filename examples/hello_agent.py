"""
hello_agent.py — The simplest possible infrarely agent (3 lines)
═══════════════════════════════════════════════════════════════════════════════
"""

import infrarely

agent = infrarely.agent("hello")
result = agent.run("2 + 2")
print(result.output)  # 4
