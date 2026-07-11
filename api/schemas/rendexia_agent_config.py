from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class BusinessHourSlot(BaseModel):
    day: str  # 'monday', 'tuesday', ...
    open: str  # '09:00'
    close: str  # '19:00'


class BookingRules(BaseModel):
    max_advance_days: int = 60
    min_notice_hours: int = 2
    max_active_bookings_per_customer: int = 3
    require_confirmation: bool = True


class RendexiaAgentConfig(BaseModel):
    persona_name: str = "Assistant"
    persona_description: Optional[str] = None
    languages: List[str] = ["fr"]
    timezone: str = "UTC"
    business_hours: List[BusinessHourSlot] = []
    booking_rules: BookingRules = BookingRules()
    greeting_message: Optional[str] = None
    system_prompt_override: Optional[str] = None
    interaction_style: Optional[str] = None  # 'professional', 'friendly', 'formal'
    escalation_policy: Optional[str] = None
    is_active: bool = True
    custom_metadata: Dict[str, Any] = {}


class RendexiaAgentConfigResponse(BaseModel):
    config: RendexiaAgentConfig
    organization_id: int
    configured: bool
