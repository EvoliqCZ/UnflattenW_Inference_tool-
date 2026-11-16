# Headless API and interactive documentation

The existing conversion service must be usable by a machine client without opening the application UI. Publish `/docs` (Swagger UI) and `/openapi.json` with usable Bearer authentication, configurable request headers, PDF/image upload, asynchronous status polling, and binary Word downloads. Preserve the current `/v1` wire contract, upload limits, authentication-before-body checks, ownership isolation, and document rendering behavior except for the separately diagnosed layout defect.

Use the existing multipart `page` upload for Swagger's file picker and retain raw PDF/PNG/JPEG requests. Document `Idempotency-Key`, `X-Unflatten-Profile`, `X-Unflatten-Output`, optional model selection and label. Word currently uses `diagnostic_docx`; do not advertise unsupported `docx` or arbitrary generation settings. Provide local Swagger assets, no external validator, no persisted API tokens, and the existing host/security middleware. One shared schema builder supplies runtime and committed OpenAPI. Document headless curl usage and a configurable client example in both README languages and an API guide. Keep the current loopback binding.

Approach: enrich the existing routes instead of introducing a second conversion API; ship pinned local Swagger assets instead of relying on browser CDN access. The requested upload/auth/download workflow defines the scope directly.

Additional user request: investigate the screenshot's nearly empty DOCX second page. Compare real IR, OOXML and rendering before changing assembly. Preserve source documents; use synthetic regression fixtures and write corrected output separately. This is independent of API documentation.
