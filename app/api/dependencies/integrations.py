from __future__ import annotations

import httpx

from app.adapters.ats import ATSAdapter
from app.adapters.crm import CRMAdapter
from app.adapters.doc_store import DocumentStoreAdapter
from app.adapters.greenhouse import GreenhouseATSAdapter
from app.adapters.mock import MockATSAdapter, MockCRMAdapter, MockDocumentStoreAdapter
from app.config import settings
from app.core.exceptions import ValidationError

_crm = MockCRMAdapter()
_ats = MockATSAdapter()
_doc_store = MockDocumentStoreAdapter()
_greenhouse_client = httpx.Client(timeout=15.0)


def get_crm_adapter() -> CRMAdapter:
    if settings.crm_provider != "mock":
        raise ValidationError("Unsupported CRM provider.", details={"provider": settings.crm_provider})
    return _crm


def get_ats_adapter() -> ATSAdapter:
    if settings.ats_provider == "mock":
        return _ats
    if settings.ats_provider == "greenhouse":
        if not settings.greenhouse_harvest_api_key:
            raise ValidationError(
                "Greenhouse ATS provider requires greenhouse_harvest_api_key.",
                details={"provider": settings.ats_provider},
            )
        return GreenhouseATSAdapter(
            api_token=settings.greenhouse_harvest_api_key,
            base_url=settings.greenhouse_base_url,
            per_page=settings.greenhouse_per_page,
            max_pages=settings.greenhouse_max_pages,
            client=_greenhouse_client,
        )
    raise ValidationError("Unsupported ATS provider.", details={"provider": settings.ats_provider})


def get_document_store_adapter() -> DocumentStoreAdapter:
    if settings.doc_store_provider != "mock":
        raise ValidationError("Unsupported document store provider.", details={"provider": settings.doc_store_provider})
    return _doc_store
