"""
Hooks attached to the Agent:
1. API call metadata logging (used by the /api-detail command)
2. system-reminder injection (appends stale file alerts and task reminders before each model request)
3. Automatic retries on API request failures (wrap_model_request)
4. Graceful fallback for unhandled tool execution errors (on_tool_execute_error)
"""
import asyncio
import dataclasses
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import ModelHTTPError, ModelAPIError
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, UserPromptPart

import classifier
import permissions
from ui.render import console, print_step

from .reminders import build_reminder_text, build_task_reminder_text


MAX_RETRIES = 3

# Trigger thresholds for task reminders
# Require >= this many turns since the last task_create / task_update call before prompting
TASK_REMINDER_TURNS_SINCE_WRITE = 6
# Require >= this many turns between task reminders to prevent spamming
TASK_REMINDER_TURNS_BETWEEN = 4

# Tools categorized as "task management" actions—invoking them resets the turns_since_write counter
_TASK_MANAGEMENT_TOOLS = {"task_create", "task_update"}


@dataclass
class ApiCall:
    """
    Metadata for a single model API call.
    Request fields are populated by before_model_request; response fields by after_model_request.
    """
    # Request side
    model: str
    messages_count: int
    # The final part of the last message sent to the model
    last_part: Any
    tools: list
    # Response side (populated by after hook)
    finish_reason: str = ""
    parts_kinds: list = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


# Cleared by the main loop before each agent.iter() turn
api_call_log: list[ApiCall] = []

hooks = Hooks()


# ---------- API Call Logging ----------

@hooks.on.before_model_request
async def _record_request(ctx, request_context):
    """
    Creates an ApiCall record before dispatching each model invocation.
    """
    msgs = list(request_context.messages)
    last_part = msgs[-1].parts[-1] if msgs and msgs[-1].parts else None
    try:
        tool_names = [t.name for t in request_context.model_request_parameters.function_tools]
    except AttributeError:
        tool_names = []
    api_call_log.append(ApiCall(
        model=request_context.model.model_name,
        messages_count=len(msgs),
        last_part=last_part,
        tools=tool_names,
    ))
    return request_context


@hooks.on.after_model_request
async def _record_response(ctx, *, request_context, response):
    """
    Populates response fields on the active ApiCall record upon completion.
    """
    if api_call_log:
        call = api_call_log[-1]
        call.finish_reason = str(response.finish_reason) if response.finish_reason else "unknown"
        call.parts_kinds = [p.part_kind for p in response.parts]
        call.input_tokens = response.usage.input_tokens
        call.output_tokens = response.usage.output_tokens
    return response


# ---------- System-Reminder Injection ----------

def _scan_task_turn_counters(messages) -> tuple[int, int]:
    """
    Performs a single reverse scan over history to compute:
    (turns since last task management tool, turns since last task_reminder).
    Exits early once both targets are found to minimize overhead on long histories.
    """
    since_mgmt = 0
    since_reminder = 0
    found_mgmt = False
    found_reminder = False
    for msg in reversed(messages):
        if isinstance(msg, ModelResponse):
            if not found_mgmt:
                for part in msg.parts:
                    if isinstance(part, ToolCallPart) and part.tool_name in _TASK_MANAGEMENT_TOOLS:
                        found_mgmt = True
                        break
                if not found_mgmt:
                    since_mgmt += 1
            if not found_reminder:
                since_reminder += 1
        elif isinstance(msg, ModelRequest) and not found_reminder:
            for part in msg.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str) and _REMINDER_SENTINELS["task"] in part.content:
                    found_reminder = True
                    break
        if found_mgmt and found_reminder:
            break
    return since_mgmt, since_reminder


def _build_file_reminder(ctx, messages) -> str | None:
    # Single-pass check for ReadFileState mutations; inspects state only, ignoring messages
    return build_reminder_text(ctx.deps.read_file_state)


def _build_task_reminder(ctx, messages) -> str | None:
    # Trigger task reminder only when both thresholds are met: prolonged inactivity and sufficient interval
    since_mgmt, since_reminder = _scan_task_turn_counters(messages)
    if since_mgmt < TASK_REMINDER_TURNS_SINCE_WRITE:
        return None
    if since_reminder < TASK_REMINDER_TURNS_BETWEEN:
        return None
    return build_task_reminder_text(ctx.deps.tasks_store)


# Registered reminder builders executed in before_model_request: (sentinel, builder) pairs.
# Sentinels are used solely for reverse-scan identification (e.g., task reminders).
_REMINDER_SENTINELS = {
    # Task reminder sentinel matches the static phrase guaranteed by the reminder builder
    "task": "The task tools haven't been used recently",
}
_REMINDER_BUILDERS = (_build_file_reminder, _build_task_reminder)


@hooks.on.before_model_request
async def _inject_reminders(ctx, request_context):
    """
    Executes each registered reminder builder sequentially, appending non-None results
    to request_context.messages. Currently handles stale file notices (via ReadFileState)
    and task reminders (via turn counters and TasksStore). New reminder types can be added
    by registering a builder without altering hook plumbing.
    """
    messages = list(request_context.messages)
    appended = []
    for builder in _REMINDER_BUILDERS:
        text = builder(ctx, messages + appended)
        if text is None:
            continue
        appended.append(ModelRequest(parts=[UserPromptPart(content=text)]))
        print_step("[dim]◇ system[/]", f"[dim]{text[:200]}[/]")
    if not appended:
        return request_context
    return dataclasses.replace(request_context, messages=messages + appended)


# ---------- API Request Retries ----------

@hooks.on.model_request
async def _retry_on_error(ctx, *, request_context, handler):
    """
    Wraps model requests with exponential backoff retries on transient errors.

    Retries are handled internally within the wrapper, leaving conversation history
    and before/after hooks unaffected.
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            return await handler(request_context)
        except ModelHTTPError as e:
            if e.status_code < 500:
                raise
            if attempt >= MAX_RETRIES:
                console.print(f"[bold red]✗ HTTP {e.status_code}: failed after {MAX_RETRIES} retries[/]")
                raise
            wait = 2 ** attempt
            console.print(
                f"[bold yellow]⟳ HTTP {e.status_code}: retrying in {wait}s "
                f"({attempt + 1}/{MAX_RETRIES})...[/]"
            )
            await asyncio.sleep(wait)
        except ModelAPIError as e:
            if attempt >= MAX_RETRIES:
                console.print(
                    f"[bold red]✗ Network connection failed after {MAX_RETRIES} retries[/]"
                )
                raise
            wait = 2 ** attempt
            console.print(
                f"[bold yellow]⟳ Network connection failed: retrying in {wait}s "
                f"({attempt + 1}/{MAX_RETRIES})...[/]"
            )
            await asyncio.sleep(wait)


# ---------- Tool Execution Permission Checks ----------

@hooks.on.tool_execute
async def _check_permission(ctx, *, call, tool_def, args, handler):
    """
    Permission gate before tool execution:
    - 'allow': invokes handler to proceed with execution.
    - 'ask': prompts user for approval.
    - 'deny': returns refusal reason as tool output for self-correction.
    In auto mode, 'ask' delegates decisions to an LLM classifier rather than prompting immediately.
    """
    decision = permissions.compute_decision(call.tool_name, args)
    if decision == "allow":
        # Approved: proceed with actual execution via handler(args)
        return await handler(args)

    # decision == "ask" in AUTO mode: delegate decision-making to the classifier
    if permissions.state.mode == permissions.AUTO:
        # Align review status with thinking and tool_call visual styles: dedicated icon/label line with indented body
        print_step("[blue]◆ auto_check[/]", f"[blue dim]Reviewing {call.tool_name} with classifier...[/]")
        verdict = await classifier.classify(ctx.messages, call.tool_name, args)

        if verdict.get("error"):
            # Fall back to interactive prompt on classifier failure; never allow on error
            print_step("[yellow]⚠ auto_check[/]", f"[yellow]{verdict['reason']}[/]")
        elif not verdict["should_block"]:
            # Classified safe: allow execution and log rationale for auditability
            print_step("[blue]✔ auto_check[/]", f"[blue dim]Allowed: {verdict['reason']}[/]")
            return await handler(args)
        else:
            # Classified hazardous: inject rejection notice back to the model with explanation
            print_step("[red]✘ auto_check[/]", f"[red]Blocked: {verdict['reason']}[/]")
            return (
                f"Security check blocked {call.tool_name}; execution skipped. Reason: {verdict['reason']}. "
                "Do not attempt to bypass this restriction. Stop and consult the user on how to proceed."
            )

    # In default/acceptEdits mode, or as fallback on classifier failure in auto mode: prompt user
    choice = await permissions.prompt_approval(call.tool_name, args)
    if choice == "once":
        return await handler(args)
    if choice == "always":
        # Add to session whitelist to suppress further prompts for this tool
        permissions.state.session_allowed.add(call.tool_name)
        return await handler(args)

    # Denied: skip execution and return refusal message to model so it pauses for user guidance
    return f"User rejected execution of {call.tool_name}. Halt current operations and await user instructions."


# ---------- Tool Execution Error Handling ----------

@hooks.on.tool_execute_error
async def _handle_tool_error(ctx, *, call, tool_def, args, error):
    """
    Prevents crash on unhandled tool exceptions; returns error output to the model for self-correction.
    """
    console.print(f"[bold red]✗ Tool {call.tool_name} failed: {error}[/]")
    return f"Tool execution failed: {type(error).__name__}: {error}"
