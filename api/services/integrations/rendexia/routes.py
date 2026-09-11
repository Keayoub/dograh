from typing import Annotated, Any, cast
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pipecat.utils.run_context import set_current_run_id

from api.db import db_client
from api.db.models import UserModel
from api.enums import OrganizationConfigurationKey, WorkflowRunMode
from api.services.auth.depends import get_user
from api.services.quota_service import authorize_workflow_run_start
from api.services.workflow.initial_context import merge_external_initial_context
from api.services.workflow.run_creation import prepare_workflow_run_inputs
from api.services.workflow.text_chat_session_service import (
    TextChatPendingTurnLostError,
    TextChatSessionExecutionError,
    TextChatSessionRevisionConflictError,
    append_text_chat_user_message,
    default_text_chat_checkpoint,
    default_text_chat_session_data,
    execute_pending_text_chat_turn,
    initialize_text_chat_session,
    normalize_text_chat_session_data,
)

from .schemas import (
    RendexiaAgentConfig,
    RendexiaAgentConfigResponse,
    RendexiaOrgProvisionRequest,
    RendexiaOrgProvisionResponse,
    RendexiaTurnRequest,
    RendexiaTurnResponse,
)

router = APIRouter(prefix="/rendexia", tags=["rendexia"])
organization_router = APIRouter(prefix="/organizations", tags=["organizations"])


def _revision_conflict_detail(exc: Any) -> dict[str, Any]:
    return {
        "message": "Text chat session revision conflict",
        "expected_revision": exc.expected_revision,
        "actual_revision": exc.actual_revision,
    }


def _extract_reply(text_session: Any) -> str:
    session_data = normalize_text_chat_session_data(text_session.session_data)
    for turn in reversed(session_data.get("turns") or []):
        if turn.get("status") != "completed":
            continue
        assistant_message = turn.get("assistant_message") or {}
        text = assistant_message.get("text")
        if text:
            return str(text)
    return ""


def _extract_suggested_action(text_session: Any) -> str | None:
    workflow_run = text_session.workflow_run
    gathered_context = workflow_run.gathered_context if workflow_run else None
    if not gathered_context:
        return None
    suggested_action = gathered_context.get("suggested_action")
    return str(suggested_action) if suggested_action is not None else None


def _state_value(state: Any) -> str:
    return state.value if hasattr(state, "value") else str(state)


async def _execute_turn(
    *,
    workflow_id: int,
    run_id: int,
    text_session: Any,
    conversation_id: str,
) -> Any:
    try:
        return await execute_pending_text_chat_turn(
            workflow_id=workflow_id,
            run_id=run_id,
            text_session=text_session,
        )
    except TextChatSessionRevisionConflictError as exc:
        raise HTTPException(
            status_code=409, detail=_revision_conflict_detail(exc)
        ) from exc
    except TextChatPendingTurnLostError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except TextChatSessionExecutionError as exc:
        logger.error(
            "Rendexia text turn execution failed for conversation {}: {}",
            conversation_id,
            exc,
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post(
    "/workflow/{workflow_uuid}/turn",
    response_model=RendexiaTurnResponse,
    summary="Execute one Rendexia assistant text turn",
    description=(
        "Submit a text-channel message. Reuse `conversation_id` to resume a "
        "session. The response fields are snake_case: `reply_text`, "
        "`conversation_id`, `workflow_run_id`, `state`, `suggested_action`, "
        "and `metadata`."
    ),
)
async def rendexia_turn(
    workflow_uuid: str,
    request: RendexiaTurnRequest,
) -> RendexiaTurnResponse:
    workflow = await db_client.get_workflow_by_uuid_unscoped(workflow_uuid)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    text_session = await db_client.get_text_session_by_conversation_id(
        workflow_uuid=workflow_uuid,
        conversation_id=request.conversation_id,
    )

    if text_session is None:
        if workflow.user_id is None:
            raise HTTPException(
                status_code=409,
                detail="Workflow has no execution owner (user_id is null)",
            )
        if workflow.organization_id is None:
            raise HTTPException(
                status_code=409,
                detail="Workflow has no organization",
            )
        workflow_id = cast(int, workflow.id)
        execution_user_id = cast(int, workflow.user_id)
        organization_id = cast(int, workflow.organization_id)

        initial_context = merge_external_initial_context(
            {
                "conversation_id": request.conversation_id,
                "tenant_id": request.tenant_id,
                "customer_phone": request.customer_phone,
                "customer_name": request.customer_name,
                "locale": request.locale,
                "channel": request.channel,
                "message": request.message,
            },
            request.initial_context,
        )
        run_inputs = await prepare_workflow_run_inputs(
            db_client,
            workflow,
            initial_context=initial_context,
            use_draft=False,
        )
        if run_inputs.definition_id is None:
            raise HTTPException(
                status_code=409,
                detail="Workflow has no published definition",
            )

        try:
            workflow_run = await db_client.create_workflow_run(
                name=(
                    f"rendexia:{request.conversation_id[:40]}-{uuid4().hex[:6].upper()}"
                ),
                workflow_id=workflow_id,
                mode=WorkflowRunMode.TEXTCHAT.value,
                user_id=execution_user_id,
                initial_context=run_inputs.initial_context,
                organization_id=organization_id,
                definition_id=run_inputs.definition_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        workflow_run_id = cast(int, workflow_run.id)
        quota_result = await authorize_workflow_run_start(
            workflow_id=workflow_id,
            organization_id=organization_id,
            workflow_run_id=workflow_run_id,
        )
        if not quota_result.has_quota:
            raise HTTPException(
                status_code=402,
                detail=quota_result.error_message or "Could not authorize workflow run",
            )

        set_current_run_id(workflow_run_id)
        workflow_run = await db_client.update_workflow_run(
            workflow_run_id,
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
            workflow_run_id,
            session_data=default_text_chat_session_data(),
            checkpoint=default_text_chat_checkpoint(),
        )
        try:
            text_session = await initialize_text_chat_session(
                run_id=workflow_run_id,
                text_session=text_session,
            )
        except TextChatSessionRevisionConflictError as exc:
            raise HTTPException(
                status_code=409, detail=_revision_conflict_detail(exc)
            ) from exc
    else:
        workflow_run_id = cast(int, text_session.workflow_run_id)
        set_current_run_id(workflow_run_id)
        try:
            text_session = await append_text_chat_user_message(
                run_id=workflow_run_id,
                text_session=text_session,
                user_text=request.message,
                expected_revision=None,
            )
        except TextChatSessionRevisionConflictError as exc:
            raise HTTPException(
                status_code=409, detail=_revision_conflict_detail(exc)
            ) from exc

    run_id = cast(int, text_session.workflow_run_id)
    workflow_id = cast(int, text_session.workflow_run.workflow_id)
    result_session = await _execute_turn(
        workflow_id=workflow_id,
        run_id=run_id,
        text_session=text_session,
        conversation_id=request.conversation_id,
    )
    workflow_run = result_session.workflow_run
    state: Any = workflow_run.state if workflow_run else "completed"

    return RendexiaTurnResponse(
        reply_text=_extract_reply(result_session),
        conversation_id=request.conversation_id,
        workflow_run_id=run_id,
        state=_state_value(state),
        suggested_action=_extract_suggested_action(result_session),
    )


@router.post(
    "/organizations",
    response_model=RendexiaOrgProvisionResponse,
    summary="Provision a Rendexia organization idempotently",
)
async def provision_rendexia_org(
    request: RendexiaOrgProvisionRequest,
) -> RendexiaOrgProvisionResponse:
    (
        organization,
        was_created,
    ) = await db_client.get_or_create_organization_by_provider_id(
        org_provider_id=request.provider_id,
        user_id=None,
    )
    return RendexiaOrgProvisionResponse(
        organization_id=cast(int, organization.id),
        provider_id=cast(str, organization.provider_id),
        already_existed=not was_created,
    )


@organization_router.get(
    "/rendexia-agent-config",
    response_model=RendexiaAgentConfigResponse,
)
async def get_rendexia_agent_config(
    user: Annotated[UserModel, Depends(get_user)],
) -> RendexiaAgentConfigResponse:
    if user.selected_organization_id is None:
        raise HTTPException(status_code=400, detail="No organization selected")
    organization_id = cast(int, user.selected_organization_id)
    config = await db_client.get_configuration(
        organization_id,
        OrganizationConfigurationKey.RENDEXIA_AGENT_CONFIG.value,
    )
    config_value = cast(dict[str, Any], config.value) if config is not None else {}
    if not config_value:
        return RendexiaAgentConfigResponse(
            config=RendexiaAgentConfig(),
            organization_id=organization_id,
            configured=False,
        )
    return RendexiaAgentConfigResponse(
        config=RendexiaAgentConfig.model_validate(config_value),
        organization_id=organization_id,
        configured=True,
    )


@organization_router.post(
    "/rendexia-agent-config",
    response_model=RendexiaAgentConfigResponse,
)
async def save_rendexia_agent_config(
    request: RendexiaAgentConfig,
    user: Annotated[UserModel, Depends(get_user)],
) -> RendexiaAgentConfigResponse:
    if user.selected_organization_id is None:
        raise HTTPException(status_code=400, detail="No organization selected")
    organization_id = cast(int, user.selected_organization_id)
    await db_client.upsert_configuration(
        organization_id,
        OrganizationConfigurationKey.RENDEXIA_AGENT_CONFIG.value,
        request.model_dump(mode="json"),
    )
    return RendexiaAgentConfigResponse(
        config=request,
        organization_id=organization_id,
        configured=True,
    )


@organization_router.delete("/rendexia-agent-config")
async def delete_rendexia_agent_config(
    user: Annotated[UserModel, Depends(get_user)],
) -> dict[str, str]:
    if user.selected_organization_id is None:
        raise HTTPException(status_code=400, detail="No organization selected")
    organization_id = cast(int, user.selected_organization_id)
    await db_client.delete_configuration(
        organization_id,
        OrganizationConfigurationKey.RENDEXIA_AGENT_CONFIG.value,
    )
    return {"message": "Rendexia agent configuration reset to defaults."}
