"""
Hooks attached to the Agent, used to capture metadata for every model API call.

The main loop clears api_call_log before each run_sync and snapshots it into
SessionState afterwards; the /api-detail command then shows the user every call
made during that turn.
"""
import asyncio
import dataclasses
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import ModelHTTPError, ModelAPIError
from pydantic_ai.messages import ModelRequest, UserPromptPart

import classifier
import permissions
from ui.render import console, print_step

from .file_state import ReadFileState
from .reminders import build_reminder_text


MAX_RETRIES = 3


@dataclass
class ApiCall:
    """
    Metadata for a single model API call. before_model_request creates the record
    and fills in the top half; after_model_request fills in the bottom half.
    """
    # request side
    model: str
    messages_count: int
    # the last part of the last message sent to the model in this call
    last_part: Any
    tools: list
    # response side (filled in by the after hook)
    finish_reason: str = ""
    parts_kinds: list = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


# the main loop clears this before each run_sync
api_call_log: list[ApiCall] = []

hooks = Hooks()


@hooks.on.before_model_request
async def _record_request(ctx, request_context):
    """
    Create an ApiCall record right before each model request goes out.
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
async def _record_response(ctx, request_context, response):
    """
    Fill in the response fields of the ApiCall created above, once the model call
    has returned.
    """
    if api_call_log:
        call = api_call_log[-1]
        call.finish_reason = str(response.finish_reason) if response.finish_reason else "unknown"
        call.parts_kinds = [p.part_kind for p in response.parts]
        call.input_tokens = response.usage.input_tokens
        call.output_tokens = response.usage.output_tokens
    return response


# ---------- API Request Retries ----------

@hooks.on.model_request
async def _retry_on_error(ctx, *, request_context, handler):
    """
    Wrap model requests with exponential backoff for retryable errors.

    Retries are handled internally; conversation history and before/after hooks remain unaffected.
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            return await handler(request_context)
        except ModelHTTPError as e:
            if e.status_code < 500:
                raise
            if attempt >= MAX_RETRIES:
                console.print(f"[bold red]✗ HTTP {e.status_code}: Failed after {MAX_RETRIES} retries[/]")
                raise
            wait = 2 ** attempt
            console.print(
                f"[bold yellow]⟳ HTTP {e.status_code}: Retrying in {wait}s "
                f"({attempt + 1}/{MAX_RETRIES})...[/]"
            )
            await asyncio.sleep(wait)
        except ModelAPIError as e:
            if attempt >= MAX_RETRIES:
                console.print(
                    f"[bold red]✗ Connection failed: Unable to connect after {MAX_RETRIES} retries[/]"
                )
                raise
            wait = 2 ** attempt
            console.print(
                f"[bold yellow]⟳ Connection failed: Retrying in {wait}s "
                f"({attempt + 1}/{MAX_RETRIES})...[/]"
            )
            await asyncio.sleep(wait)


# ---------- Tool Execution Fallback ----------

@hooks.on.tool_execute_error
async def _handle_tool_error(ctx, *, call, tool_def, args, error):
    """
    Catch unhandled tool exceptions and return the error message as a tool result
    for the model to self-correct, preventing process crashes.
    """
    console.print(f"[bold red]✗ Tool {call.tool_name} failed: {error}[/]")
    return f"Tool execution failed: {type(error).__name__}: {error}"


# ---------- Tool Execution Permission Check ----------

@hooks.on.tool_execute
async def _check_permission(ctx, *, call, tool_def, args, handler):
    """
    Permission gate before tool execution.

    Proceeds if allowed, prompts for approval if required, and returns the denial
    reason as the tool result if rejected so the model can self-correct.
    In auto mode, pending calls are evaluated by an LLM classifier before prompting.
    """
    decision = permissions.compute_decision(call.tool_name, args)
    if decision == "allow":
        # Authorized: execute the tool via handler
        return await handler(args)

    # In auto mode, delegate pending approvals to the LLM classifier
    if permissions.state.mode == permissions.AUTO:
        # Match step formatting (icon + label header, indented body) using blue for classifier status
        print_step("[blue]◆ auto_check[/]", f"[blue dim]Evaluating {call.tool_name} via LLM...[/]")
        verdict = await classifier.classify(ctx.messages, call.tool_name, args)

        if verdict.get("error"):
            # Fail-closed: fall back to manual approval on evaluation error
            print_step("[yellow]⚠ auto_check[/]", f"[yellow]{verdict['reason']}[/]")
        elif not verdict["should_block"]:
            # Approved: execute tool and log reason for auditability
            print_step("[blue]✔ auto_check[/]", f"[blue dim]Allowed: {verdict['reason']}[/]")
            return await handler(args)
        else:
            # Blocked: return denial message with reason to model
            print_step("[red]✘ auto_check[/]", f"[red]Blocked: {verdict['reason']}[/]")
            return (
                f"Security check blocked execution of {call.tool_name}. Reason: {verdict['reason']}. "
                "Do not attempt to bypass this restriction. Pause operations and notify the user for guidance."
            )

    # Manual approval fallback (default/acceptEdits mode, or classifier errors in auto mode)
    choice = await permissions.prompt_approval(call.tool_name, args)
    if choice == "once":
        return await handler(args)
    if choice == "always":
        # Add to session allowlist to skip future prompts in this session
        permissions.state.session_allowed.add(call.tool_name)
        return await handler(args)

    # Denied: return rejection notice as tool output so the model pauses for user guidance
    return (
        f"Execution of {call.tool_name} was denied by the user. "
        "Pause operations and await further instructions."
    )


# ---------- System-Reminder Injection ----------

@hooks.on.before_model_request
async def _inject_reminders(ctx, request_context):
    """
    Appends a system-reminder before each model request if any files in ReadFileState
    were modified externally. Once re-read by the model, record() updates the timestamp
    so subsequent hooks omit the reminder automatically.
    """
    state: ReadFileState = ctx.deps
    text = build_reminder_text(state)
    if text is None:
        return request_context

    reminder = ModelRequest(parts=[UserPromptPart(content=text)])
    new_messages = list(request_context.messages) + [reminder]
    print_step("[dim]◇ system[/]", f"[dim]{text[:200]}[/]")
    return dataclasses.replace(request_context, messages=new_messages)
