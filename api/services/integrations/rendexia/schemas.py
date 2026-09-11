from typing import Any, Literal

from pydantic import BaseModel, Field


class BusinessHourSlot(BaseModel):
    day: str
    open: str
    close: str


class BookingRules(BaseModel):
    max_advance_days: int = 60
    min_notice_hours: int = 2
    max_active_bookings_per_customer: int = 3
    require_confirmation: bool = True


class RendexiaAgentConfig(BaseModel):
    persona_name: str = "Assistant"
    persona_description: str | None = None
    languages: list[str] = Field(default_factory=lambda: ["fr"])
    timezone: str = "UTC"
    business_hours: list[BusinessHourSlot] = Field(default_factory=list)
    booking_rules: BookingRules = Field(default_factory=BookingRules)
    greeting_message: str | None = None
    system_prompt_override: str | None = None
    interaction_style: Literal["professional", "friendly", "formal"] | None = None
    escalation_policy: str | None = None
    is_active: bool = True
    custom_metadata: dict[str, Any] = Field(default_factory=dict)


class RendexiaAgentConfigResponse(BaseModel):
    config: RendexiaAgentConfig
    organization_id: int
    configured: bool


class RendexiaTurnRequest(BaseModel):
    message: str
    conversation_id: str
    tenant_id: str
    customer_phone: str | None = None
    customer_name: str | None = None
    locale: str = "fr"
    channel: Literal["whatsapp", "sms", "web"] = "whatsapp"
    initial_context: dict[str, Any] | None = None


class RendexiaTurnResponse(BaseModel):
    """Snake-case response contract consumed by Rendexia."""

    reply_text: str
    conversation_id: str
    workflow_run_id: int | None = None
    state: str = "completed"
    suggested_action: str | None = None
    metadata: dict[str, Any] | None = None


class RendexiaOrgProvisionRequest(BaseModel):
    tenant_id: str
    tenant_name: str
    provider_id: str
    plan: str = "rendexia"


class RendexiaOrgProvisionResponse(BaseModel):
    organization_id: int
    provider_id: str
    already_existed: bool
