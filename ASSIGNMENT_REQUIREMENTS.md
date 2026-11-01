# INTERN TRAINING
# Function Calling and AI Agents

*Converted from the instructor's PowerPoint. The original slide order and terminology are preserved. Text content is transcribed below; slide visuals and animations are not embedded.*

---

## Slide 1 — INTERN TRAINING

**Function Calling and AI Agents**

---

## Slide 2 — What is a Large Language Model?

**Input**

Text you give it: instructions, question, documents. Split into tokens (word pieces).

**Model**

A neural network trained on huge amounts of text to predict the next token, again and again.

**Output**

Fluent text, one token at a time. It can also be structured JSON if asked.

**Analogy:** an extremely well-read assistant who writes by predicting what comes next, with no live access to the world.

---

## Slide 3 — What an LLM cannot do alone

**No live data**

Knowledge stops at its training cutoff. It cannot see your database or today's news.

**No actions**

It writes text. It cannot send an email, charge a card or check stock.

**No memory**

Each call is independent. Anything it should remember must be sent again.

**Can be wrong**

Fluent does not mean true. Without a source it may state guesses as facts.

A chatbot can explain an inventory system. It cannot check inventory unless your application gives it a safe action path.

---

## Slide 4 — What is an agent?

An AI agent is a goal-oriented system that can make decisions, use tools, manage state, and perform actions to accomplish a task.

**Goal-oriented**

Works toward an outcome you define, not just one reply.

**Makes decisions**

Chooses the next step: which tool, which branch, when to stop.

**Uses tools & acts**

Calls tools and performs real actions: search, query, write, send.

**Manages state**

Keeps the goal, progress and results across many steps.

---

## Slide 5 — Anatomy of an agent

**Model**

The LLM that reasons and chooses actions.

**Instructions**

System prompt: role, rules, tone, when to stop.

**Tools**

External capabilities that allow to act and communicate with the world.

**Memory**

Conversation history plus optional long-term storage.

**Loop and state**

Code that repeats steps and tracks progress.

**Guardrails**

Limits, validation and approvals around every action.

---

## Slide 6 — Agentic nature 1: Execution tools

**What they are**

Tools that let the agent act and observe beyond text. The model decides, the tool does the work.

**Function calling**

Structured calls to your own APIs. E.g. check_inventory, create_ticket. Covered in depth next.

**Searching tools**

Fetch fresh facts. E.g. web, news, Wikipedia or arXiv search.

**Code & compute**

Run code for exact results. E.g. Python sandbox, SQL, calculator.

**RAG**

Retrieve relevant chunks from your documents, then answer with citations. E.g. vector DB over policies.

**System & environment**

Interact with the machine and apps. E.g. read files, shell, browser, calendar.

---

## Slide 7 — Agentic nature 2: Orchestration mechanisms

**Planning**

Break the goal into steps, then revise the plan when evidence changes.

**Routing**

Choose the right tool, branch or specialist agent for each step.

**State & memory**

Track goal, progress, evidence and retries across steps.

**Control flow**

Loops, branches, parallel calls, retries and fallbacks.

**Human in the loop**

Pause for approval before risky or irreversible actions.

**Guardrails & stop rules**

Validate outputs, limit steps and stop when done, blocked or unsafe.

---

## Slide 8 — Agentic nature 3: Meta-tools

**What they are**

Tools that act on the agent's own process, not on the outside world.

**Tool discovery**

Search a large tool catalog and load only what is needed.

**Planning / to-do**

Write and update a task list the agent follows and checks off.

**Delegation**

Spawn a specialist sub-agent and collect its result.

**Critique & memory**

Review its own answer and save lessons or notes for later steps.

**Ask user & finish**

Request missing information, or declare the goal complete.

---

## Slide 9 — Agent Architectures and Reasoning Patterns

| Type | Core idea | Use when |
|---|---|---|
| ReAct | Alternate Reason, Act, Observe in a loop | General tool-using tasks |
| Reflexion | Learn from past failures via written self-feedback (verbal reinforcement) | Retryable tasks, coding |
| Plan-and-Execute | Plan all steps first, execute, replan if needed | Long multi-step jobs |
| Tree of Thoughts | Explore several reasoning branches, score, keep the best | Puzzles, hard decisions |
| Multi-agent systems | Specialist agents collaborate under a supervisor | Complex, splittable work |

---

## Slide 10 — Function Calling

### The old way: brittle routing code

```python
if "cancel" and "order":
    cancel_order(order_id)
elif "track":
    track_shipment(order_id)
elif "stock":
    check_inventory(product, size)
else:
    return "not sure"
```

- One message can hold several intents.
- Ambiguous wording breaks keyword rules.
- Every new capability adds branches.
- Extracting parameters becomes its own NLP project.

---

## Slide 11 — Function calling: the model proposes the route

1. **Request** — User asks in natural language.
2. **Definitions** — You send tool names, descriptions and JSON schemas.
3. **Tool call** — Model returns a function name plus structured arguments.
4. **Runtime** — Your code validates, authorizes, runs, returns output.
5. **Answer** — Model replies in language grounded in the result.

The model decides what to ask for. The application decides what to trust and execute.

---

## Slide 12 — The schema is the contract

```json
{
  "type": "function",
  "name": "check_inventory",
  "description": "Check stock for a product",
  "parameters": {
    "type": "object",
    "properties": {
      "product_name": {"type": "string"},
      "size": {"type": "string"}
    },
    "required": ["product_name"]
  }
}
```

- **name:** the operation it can request
- **description:** guidance on when to use it
- **parameters:** the JSON shape your code expects

---

## Slide 13 — A tool request is not a result

```json
{
  "type": "function_call",
  "name": "check_inventory",
  "arguments": {
    "product_name": "blue hoodie",
    "size": "M"
  },
  "call_id": "call_123"
}
```

- **name** identifies the operation
- **arguments** follow the schema
- **call_id** links the later output to this request

No inventory check has happened yet. This is only the model's proposal.

---

## Slide 14 — Your application owns the execution loop

1. **Validate** — Parse the JSON and enforce the declared schema.
2. **Authorize** — Check identity, permissions, limits and business rules.
3. **Execute** — Call the API, database or internal service.
4. **Return** — Send success or error back with the call ID.
5. **Respond** — Model explains the verified result.

Errors become structured observations the model can react to. Sensitive actions can require confirmation. Every call can be logged and audited.

---

## Slide 15 — One message, several calls

> "I ordered the blue hoodie, got the red one, and need the blue one by Friday. Can you swap it and upgrade shipping?"

**lookup_order**

Find the order from customer context.

**create_exchange**

Wrong item to desired item.

**upgrade_shipping**

Meet the Friday deadline.

The model parses intent. Your system controls order, policy and approvals.

---

## Slide 16 — Function calling, tool use and MCP

**Function calling**

The model outputs a structured function request in a provider-specific schema.

**Tool use**

The wider behavior: choose tools, chain calls, observe results, decide when to stop. This is the agent loop.

**MCP**

Model Context Protocol: a standard way to discover and expose tools across apps and models.

**Function calling is the request. Tool use is the loop. MCP makes tools portable.**

---

## Slide 17 — Why use an agent framework?

You can write the loop yourself in about 30 lines. Frameworks add what you would otherwise rebuild for every project.

**State and memory**

Track progress, save checkpoints, resume after a crash.

**Orchestration**

Branching, retries, parallel steps, hand-offs between agents.

**Observability**

Traces of every model call and tool call for debugging.

**Safety hooks**

Guardrails, human approval pauses, limits.

---

## Slide 18 — Frameworks compared: mental model

| Framework | Mental model | Best for |
|---|---|---|
| LangGraph | Stateful graph of nodes and edges | Long-running, controlled workflows |
| CrewAI | Team of agents with roles and goals | Role-based collaboration, quick start |
| OpenAI Agents SDK | Agents, tools and hand-offs | Fast single or multi-agent apps |
| Microsoft Agent Framework | Successor to AutoGen and Semantic Kernel | Azure and .NET enterprise teams |
| Claude Agent SDK | The harness behind Claude Code | Coding and file-working agents |

---

## Slide 19 — LangGraph: agents as a graph

- **State:** a shared object passed between steps
- **Nodes:** functions or LLM calls that update state
- **Edges:** fixed or conditional routes between nodes
- **Checkpoints:** save state to pause, resume or replay
- **Human in the loop:** wait for approval at any node

---

## Slide 20 — The mechanism is simple. Production is not.

**Tool overload**

Too many tools raise cost and wrong-tool choices.

**Ambiguous input**

Ask a clarifying question before guessing arguments.

**Security**

Permission checks, validation, rate limits, prompt-injection defense.

**Errors**

Timeouts and failed APIs need a recovery path.

A demo needs a schema. A product needs an execution layer.

---

## Slide 21 — A practical checklist

- Narrow tools with clear names.
- Descriptions that say when to use each tool.
- Strict schemas, validated again in code.
- Separate model choice from execution authority.
- Ask for clarification when arguments are missing.
- Log requests, outputs, latency and failures.
- Handle timeouts and API errors explicitly.
- Test risky flows with adversarial prompts.

**Rule of thumb:** let the model propose; let your application dispose.

---

## Slide 22 — Task: Build an Agentic System with LangGraph

### Core Requirements

- Agentic system on any practical domain with a clear goal.
- Use LangGraph: branches, loops and state, no hard-coded linear steps.
- At least 5 tools from execution-tool categories (function calling, search, code & compute, RAG, system & environment).
- Include meta-tool: plan/to-do, ask user, critique, tool selection or delegation.
- Keep task state, validate every tool output, cite evidence, briefly explain each decision.

### Deliverables

- Runnable source code + clear setup instructions.
- README: workflow, agent pattern, tools, state, stopping rules, retries, fallbacks.
- Architecture diagram (planner → tools → memory → validation → decision).
- One-page evaluation report (success rate, failures, limitations, improvements, agentic behavior shown).

**Build an agent for a real problem of your choice. It must show agentic behavior autonomy, planning, tools, memory, self-correction.**

---

## Slide 23 — Task: Reliability and agentic evidence

Stopping rules, retries and tests turn your agent into one that can be trusted. Show the agentic behavior in logs.

**Stop when**

Goal complete, information missing, risk too high, or max steps reached.

**Retries**

Max 2. Each retry uses a corrected query or a different valid approach.

**Unreliable tests**

Missing info, conflicting sources, outdated results, tool timeouts, prompt injection.

**Agentic evidence**

Logs show a plan made and revised, tool choice reasons, a reflection after failure, state updates.
