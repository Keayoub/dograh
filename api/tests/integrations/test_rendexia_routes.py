from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.services.auth.depends import get_user
from api.services.integrations.rendexia.routes import organization_router, router
from api.services.integrations.rendexia.schemas import (
    BusinessHourSlot,
    RendexiaAgentConfig,
)


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.include_router(organization_router)
    app.dependency_overrides[get_user] = lambda: SimpleNamespace(
        id=7,
        selected_organization_id=11,
    )
    return app


def _text_session(*, reply: str = "Votre réservation est confirmée."):
    workflow_run = SimpleNamespace(
        id=501,
        workflow_id=33,
        state="running",
        gathered_context={"suggested_action": "send_confirmation"},
    )
    return SimpleNamespace(
        workflow_run_id=workflow_run.id,
        workflow_run=workflow_run,
        session_data={
            "turns": [
                {
                    "status": "completed",
                    "assistant_message": {"text": reply},
                }
            ]
        },
    )


def test_rendexia_routes_and_snake_case_contract_are_in_openapi():
    from api.app import app

    schema = app.openapi()

    assert "/api/v1/rendexia/organizations" in schema["paths"]
    assert "/api/v1/rendexia/workflow/{workflow_uuid}/turn" in schema["paths"]
    assert "/api/v1/organizations/rendexia-agent-config" in schema["paths"]
    response_schema = schema["components"]["schemas"]["RendexiaTurnResponse"]
    assert set(response_schema["properties"]) == {
        "reply_text",
        "conversation_id",
        "workflow_run_id",
        "state",
        "suggested_action",
        "metadata",
    }


def test_rendexia_agent_config_uses_independent_mutable_defaults():
    first = RendexiaAgentConfig()
    second = RendexiaAgentConfig()

    first.languages.append("en")
    first.business_hours.append(
        BusinessHourSlot(day="monday", open="09:00", close="17:00")
    )
    first.custom_metadata["source"] = "test"

    assert second.languages == ["fr"]
    assert second.business_hours == []
    assert second.custom_metadata == {}


def test_provision_organization_preserves_idempotent_response_contract():
    client = TestClient(_make_app())

    with patch("api.services.integrations.rendexia.routes.db_client") as mock_db:
        mock_db.get_or_create_organization_by_provider_id = AsyncMock(
            return_value=(
                SimpleNamespace(id=42, provider_id="tenant-123"),
                False,
            )
        )
        response = client.post(
            "/rendexia/organizations",
            json={
                "tenant_id": "tenant-123",
                "tenant_name": "Rendexia Tenant",
                "provider_id": "tenant-123",
                "plan": "rendexia",
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "organization_id": 42,
        "provider_id": "tenant-123",
        "already_existed": True,
    }
    mock_db.get_or_create_organization_by_provider_id.assert_awaited_once_with(
        org_provider_id="tenant-123",
        user_id=None,
    )


def test_existing_conversation_returns_documented_snake_case_mapping():
    client = TestClient(_make_app())
    existing_session = _text_session(reply="")
    appended_session = _text_session(reply="")
    result_session = _text_session()

    with (
        patch("api.services.integrations.rendexia.routes.db_client") as mock_db,
        patch(
            "api.services.integrations.rendexia.routes.append_text_chat_user_message",
            new=AsyncMock(return_value=appended_session),
        ) as append_message,
        patch(
            "api.services.integrations.rendexia.routes.execute_pending_text_chat_turn",
            new=AsyncMock(return_value=result_session),
        ) as execute_turn,
    ):
        mock_db.get_workflow_by_uuid_unscoped = AsyncMock(
            return_value=SimpleNamespace(id=33)
        )
        mock_db.get_text_session_by_conversation_id = AsyncMock(
            return_value=existing_session
        )

        response = client.post(
            "/rendexia/workflow/workflow-uuid/turn",
            json={
                "message": "Je souhaite réserver.",
                "conversation_id": "tenant-123:+15551234567",
                "tenant_id": "tenant-123",
                "customer_phone": "+15551234567",
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "reply_text": "Votre réservation est confirmée.",
        "conversation_id": "tenant-123:+15551234567",
        "workflow_run_id": 501,
        "state": "running",
        "suggested_action": "send_confirmation",
        "metadata": None,
    }
    append_message.assert_awaited_once_with(
        run_id=501,
        text_session=existing_session,
        user_text="Je souhaite réserver.",
        expected_revision=None,
    )
    execute_turn.assert_awaited_once_with(
        workflow_id=33,
        run_id=501,
        text_session=appended_session,
    )


def test_new_conversation_authorizes_quota_before_session_creation():
    client = TestClient(_make_app())
    workflow = SimpleNamespace(
        id=33,
        user_id=7,
        organization_id=11,
        released_definition=SimpleNamespace(id=77),
    )

    with (
        patch("api.services.integrations.rendexia.routes.db_client") as mock_db,
        patch(
            "api.services.integrations.rendexia.routes.authorize_workflow_run_start",
            new=AsyncMock(
                return_value=SimpleNamespace(
                    has_quota=False,
                    error_message="Quota exhausted",
                )
            ),
        ) as authorize,
    ):
        mock_db.get_workflow_by_uuid_unscoped = AsyncMock(return_value=workflow)
        mock_db.get_text_session_by_conversation_id = AsyncMock(return_value=None)
        mock_db.create_workflow_run = AsyncMock(return_value=SimpleNamespace(id=501))
        mock_db.ensure_workflow_run_text_session = AsyncMock()

        response = client.post(
            "/rendexia/workflow/workflow-uuid/turn",
            json={
                "message": "Bonjour",
                "conversation_id": "tenant-123:+15551234567",
                "tenant_id": "tenant-123",
                "initial_context": {
                    "booking_id": "booking-1",
                    "mps_correlation_id": "caller-controlled",
                },
            },
        )

    assert response.status_code == 402
    assert response.json() == {"detail": "Quota exhausted"}
    create_call = mock_db.create_workflow_run.await_args
    assert create_call is not None
    create_kwargs = create_call.kwargs
    assert create_kwargs["definition_id"] == 77
    assert create_kwargs["organization_id"] == 11
    assert create_kwargs["initial_context"]["booking_id"] == "booking-1"
    assert "mps_correlation_id" not in create_kwargs["initial_context"]
    authorize.assert_awaited_once_with(
        workflow_id=33,
        organization_id=11,
        workflow_run_id=501,
    )
    mock_db.ensure_workflow_run_text_session.assert_not_awaited()


def test_save_agent_config_is_scoped_to_selected_organization():
    client = TestClient(_make_app())

    with patch("api.services.integrations.rendexia.routes.db_client") as mock_db:
        mock_db.upsert_configuration = AsyncMock()
        response = client.post(
            "/organizations/rendexia-agent-config",
            json={
                "persona_name": "Concierge",
                "languages": ["fr", "en"],
                "timezone": "America/Toronto",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["organization_id"] == 11
    assert payload["configured"] is True
    assert payload["config"]["persona_name"] == "Concierge"
    mock_db.upsert_configuration.assert_awaited_once()
    upsert_call = mock_db.upsert_configuration.await_args
    assert upsert_call is not None
    assert upsert_call.args[0] == 11
