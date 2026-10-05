"""
Hooks attached to the Agent, used to capture metadata for every model API call.

The main loop clears api_call_log before each run_sync and snapshots it into
SessionState afterwards; the /api-detail command then shows the user every call
made during that turn.
"""
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.capabilities import Hooks


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
