"""Public text-turn endpoint for Rendexia WhatsApp/SMS/text messaging.

Exposes:
    POST /api/v1/rendexia/workflow/{workflow_uuid}/turn

This endpoint is intentionally **unauthenticated** — the ``workflow_uuid``
itself acts as a shared secret between Rendexia and Dograh.  No API key or
session cookie is required, making it easy to call from a WhatsApp webhook.

Session lifecycle
-----------------
Each conversation is keyed on ``conversation_id`` (= ``{tenant_id}:{customer_phone}``).
On the first turn a new WorkflowRun + TextSession is created and tagged with
``rendexia_conversation_id`` in its annotations so it can be re-found on
subsequent turns.  Subsequent turns append a user message and execute the
next assistant turn.
"""

from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from loguru import logger
from pipecat.utils.run_context import set_current_run_id
from pydantic import BaseModel

from api.db import db_client
from api.enums import WorkflowRunMode
from api.services.workflow.text_chat_session_service import (
    TextChatPendingTurnLostError,
    TextChatSessionExecutionError,
    TextChatSessionRevisionConflictError,
    append_text_chat_user_message,
    default_text_chat_checkpoint,
    default_text_chat_session_data,
    execute_pending_text_chat_turn,
    initialize_text_chat_session,
)

router = APIRouter(prefix="/rendexia", tags=["rendexia"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class RendexiaTurnRequest(BaseModel):
    message: str
    conversation_id: str  # "{tenant_id}:{customer_phone}" — used as session key
    tenant_id: str
    customer_phone: Optional[str] = None
    customer_name: Optional[str] = None
    locale: str = "fr"
    channel: str = "whatsapp"  # whatsapp | sms | web
    initial_context: Optional[Dict[str, Any]] = None  # booking context, persona config, …


class RendexiaTurnResponse(BaseModel):
    reply_text: str
    conversation_id: str
    workflow_run_id: Optional[int] = None
    state: str = "completed"
    suggested_action: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_state_value(state: Any) -> str:
    return state.value if hasattr(state, "value") else str(state)


def _extract_reply(session) -> str:
    """Extract the last assistant message from session turns.

    The authoritative location is ``session_data.turns[-1].assistant_message.text``
    which is set by ``execute_pending_text_chat_turn``.  Falls back to scanning
    turns in reverse for any completed turn with an assistant message.
    """
    from api.services.workflow.text_chat_session_service import (
        normalize_text_chat_session_data,
    )

    session_data = normalize_text_chat_session_data(session.session_data)
    turns = session_data.get("turns") or []
    for turn in reversed(turns):
        if turn.get("status") == "completed":
            assistant_msg = turn.get("assistant_message")
            if assistant_msg:
                text = assistant_msg.get("text", "")
                if text:
                    return str(text)
    return ""


def _extract_suggested_action(session) -> Optional[str]:
    """Return suggested_action from gathered_context if set by the workflow."""
    workflow_run = session.workflow_run
    if workflow_run is None:
        return None
    gathered = workflow_run.gathered_context or {}
    return gathered.get("suggested_action")


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/workflow/{workflow_uuid}/turn",
    response_model=RendexiaTurnResponse,
    summary="Rendexia — execute one text/WhatsApp turn",
    description=(
        "Submit a user message and receive the agent reply.  "
        "The endpoint is stateless from the caller's perspective: pass the same "
        "``conversation_id`` on every turn and the backend will resume the right session."
    ),
)
async def rendexia_turn(
    workflow_uuid: str,
    request: RendexiaTurnRequest,
) -> RendexiaTurnResponse:
    # ------------------------------------------------------------------
    # 1. Resolve the workflow (unscoped — uuid is the shared secret)
    # ------------------------------------------------------------------
    workflow = await db_client.get_workflow_by_uuid_unscoped(workflow_uuid)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    # ------------------------------------------------------------------
    # 2. Find or create a text-chat session for this conversation
    # ------------------------------------------------------------------
    existing_session = await db_client.get_text_session_by_conversation_id(
        workflow_uuid=workflow_uuid,
        conversation_id=request.conversation_id,
    )

    if existing_session is None:
        # ----------------------------------------------------------------
        # New conversation — create a WorkflowRun + TextSession
        # ----------------------------------------------------------------
        session_name = f"rendexia:{request.conversation_id[:40]}-{uuid4().hex[:6].upper()}"

        # create_workflow_run requires a user_id, so we pull it from the workflow owner.
        execution_user_id = workflow.user_id
        if execution_user_id is None:
            raise HTTPException(
                status_code=409,
                detail="Workflow has no execution owner (user_id is null)",
            )

        merged_initial_context = {
            **(request.initial_context or {}),
            "conversation_id": request.conversation_id,
            "tenant_id": request.tenant_id,
            "customer_phone": request.customer_phone,
            "customer_name": request.customer_name,
            "locale": request.locale,
            "channel": request.channel,
            "message": request.message,
        }

        try:
            workflow_run = await db_client.create_workflow_run(
                name=session_name,
                workflow_id=workflow.id,
                mode=WorkflowRunMode.TEXTCHAT.value,
                user_id=execution_user_id,
                initial_context=merged_initial_context,
                organization_id=workflow.organization_id,
                use_draft=False,  # use the published definition for production traffic
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        set_current_run_id(workflow_run.id)

        # Tag the run so we can find it again on subsequent turns
        await db_client.update_workflow_run(
            workflow_run.id,
            annotations={
                "rendexia_conversation_id": request.conversation_id,
                "rendexia": {
                    "tenant_id": request.tenant_id,
                    "customer_phone": request.customer_phone,
                    "customer_name": request.customer_name,
                    "channel": request.channel,
                    "locale": request.locale,
                },
            },
        )

        text_session = await db_client.ensure_workflow_run_text_session(
            workflow_run.id,
            session_data=default_text_chat_session_data(),
            checkpoint=default_text_chat_checkpoint(),
        )

        try:
            text_session = await initialize_text_chat_session(
                run_id=workflow_run.id,
                text_session=text_session,
            )
        except TextChatSessionRevisionConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Text chat session revision conflict",
                    "expected_revision": exc.expected_revision,
                    "actual_revision": exc.actual_revision,
                },
            ) from exc

    else:
        # ----------------------------------------------------------------
        # Existing conversation — append the new user message
        # ----------------------------------------------------------------
        text_session = existing_session
        set_current_run_id(text_session.workflow_run_id)

        try:
            text_session = await append_text_chat_user_message(
                run_id=text_session.workflow_run_id,
                text_session=text_session,
                user_text=request.message,
                expected_revision=None,  # last-write-wins for public channel
            )
        except TextChatSessionRevisionConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Text chat session revision conflict",
                    "expected_revision": exc.expected_revision,
                    "actual_revision": exc.actual_revision,
                },
            ) from exc

    # ------------------------------------------------------------------
    # 3. Execute the pending assistant turn
    # ------------------------------------------------------------------
    run_id = text_session.workflow_run_id
    workflow_id = text_session.workflow_run.workflow_id

    try:
        result_session = await execute_pending_text_chat_turn(
            workflow_id=workflow_id,
            run_id=run_id,
            text_session=text_session,
        )
    except TextChatSessionRevisionConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Text chat session revision conflict",
                "expected_revision": exc.expected_revision,
                "actual_revision": exc.actual_revision,
            },
        ) from exc
    except TextChatPendingTurnLostError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except TextChatSessionExecutionError as exc:
        logger.error(
            "Rendexia text turn execution failed for conversation {}: {}",
            request.conversation_id,
            exc,
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # ------------------------------------------------------------------
    # 4. Build and return the response
    # ------------------------------------------------------------------
    reply_text = _extract_reply(result_session)
    workflow_run = result_session.workflow_run

    return RendexiaTurnResponse(
        reply_text=reply_text,
        conversation_id=request.conversation_id,
        workflow_run_id=run_id,
        state=_get_state_value(workflow_run.state) if workflow_run else "completed",
        suggested_action=_extract_suggested_action(result_session),
    )


# ---- Org provisioning -------------------------------------------------------


class RendexiaOrgProvisionRequest(BaseModel):
    tenant_id: str  # Rendexia tenant UUID
    tenant_name: str
    provider_id: str  # unique external ID = tenant_id
    plan: str = "rendexia"  # plan slug


class RendexiaOrgProvisionResponse(BaseModel):
    organization_id: int
    provider_id: str
    already_existed: bool


@router.post(
    "/organizations",
    response_model=RendexiaOrgProvisionResponse,
    summary="Rendexia — idempotent org provisioning",
    description=(
        "Create or return the Dograh organization for a Rendexia tenant. "
        "Uses provider_id = tenant_id for deduplication. Safe to call multiple times."
    ),
)
async def provision_rendexia_org(
    request: RendexiaOrgProvisionRequest,
) -> RendexiaOrgProvisionResponse:
    """
    Idempotent: create or return the Dograh organization for a Rendexia tenant.
    Uses provider_id = tenant_id for deduplication.
    """
    organization, was_created = await db_client.get_or_create_organization_by_provider_id(
        org_provider_id=request.provider_id,
        user_id=0,  # system provisioning — no owning user
    )
    return RendexiaOrgProvisionResponse(
        organization_id=organization.id,
        provider_id=organization.provider_id,
        already_existed=not was_created,
    )
