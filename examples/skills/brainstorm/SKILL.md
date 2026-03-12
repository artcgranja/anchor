---
name: brainstorm
description: Guide the agent through a structured brainstorming process to explore ideas before implementation
activation: on_demand
tags: [creative, planning]
---

# Brainstorming Skill

Help the user explore ideas through structured dialogue before jumping to implementation.

## Process

1. **Understand the goal** -- Ask what the user wants to achieve and why
2. **Ask clarifying questions** -- One at a time, prefer multiple choice when possible
3. **Propose 2-3 approaches** -- Present options with trade-offs and your recommendation
4. **Summarize** -- Capture the chosen approach, key decisions, and next steps

## Guidelines

- One question per message -- don't overwhelm
- Prefer multiple choice questions over open-ended
- Always propose alternatives before settling on an approach
- Apply YAGNI ruthlessly -- remove unnecessary complexity
- Save the result when the brainstorming session concludes

## Using the Tool

After the brainstorming session, use `save_brainstorm_result` to persist the outcome:
- **title**: A short name for the session
- **summary**: The chosen approach and key decisions
- **approaches**: All approaches that were considered with trade-offs
