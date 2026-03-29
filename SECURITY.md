# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in InfraRely, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, please email: **security@infrarely.dev**

### What to include

- A description of the vulnerability
- Steps to reproduce the issue
- Affected versions
- Any potential impact assessment

### Response timeline

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 5 business days
- **Fix or mitigation**: Dependent on severity, typically within 30 days

### Severity levels

| Level    | Description                                    | Target resolution |
| -------- | ---------------------------------------------- | ----------------- |
| Critical | Remote code execution, data exfiltration       | 7 days            |
| High     | Authentication bypass, privilege escalation    | 14 days           |
| Medium   | Information disclosure, denial of service      | 30 days           |
| Low      | Minor issues with limited impact               | Next release      |

## Security Features

InfraRely includes built-in security primitives:

- **Input sanitization** — Prompt injection detection and mitigation
- **Tool execution sandbox** — Resource-limited tool execution
- **Key rotation** — Automated API key lifecycle management
- **Compliance logging** — Auditable action trails
- **Permission policies** — Scope-based agent permissions
- **Circuit breakers** — Automatic failure isolation

## Best Practices

When using InfraRely in production:

1. Always set `INFRARELY_API_KEY` via environment variables, never hardcode
2. Enable tool validation (`tool_validation_enabled: true`)
3. Configure appropriate `SecurityPolicy` for your agents
4. Use `ResourceIsolation` for multi-tenant deployments
5. Review compliance logs regularly
6. Keep InfraRely updated to the latest version

## Disclosure Policy

We follow coordinated disclosure. Reporters will be credited (unless they prefer anonymity) once the issue is resolved and a patch is released.
