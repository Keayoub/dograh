from api.services.integrations.base import IntegrationPackageSpec
from api.services.integrations.registry import register_package

from .routes import organization_router, router

PACKAGE = register_package(
    IntegrationPackageSpec(
        name="rendexia",
        routers=(router, organization_router),
    )
)

__all__ = ["PACKAGE"]
