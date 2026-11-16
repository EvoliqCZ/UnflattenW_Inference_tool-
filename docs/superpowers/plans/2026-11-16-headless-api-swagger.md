# Headless API and Swagger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Make the existing conversion API discoverable and testable, document headless integration, and diagnose/fix the separately reported Word pagination defect.

**Architecture:** Preserve existing handlers and enrich their OpenAPI contract in a focused module. Serve same-origin Swagger assets and schema through the existing secured app. Diagnose renderer changes independently.

**Tech Stack:** Python/FastAPI, OpenAPI 3.1, Swagger UI, pytest, existing TypeScript frontend and Docker deployment.

**Spec:** `docs/superpowers/specs/2026-11-16-headless-api-design.md`

## Global Constraints

- Preserve raw PDF/PNG/JPEG and multipart `page` input contracts.
- Authenticate before reading upload bytes; preserve limits and principal ownership.
- Use `diagnostic_docx` for Word and `word-core-v0` profile.
- Do not embed or commit actual tokens or user documents.
- Use local pinned assets, no external validator, no persisted Swagger tokens.
- Preserve loopback binding and existing worker/model data.

## Review Focus

- A fresh Swagger browser can authorize, select a PDF and send accepted multipart data.
- Required headers and supported output/model values match runtime validation.
- Downloads are binary attachments and require ownership/authentication.
- Headless clients handle failed/expired jobs and stop polling at a deadline.
- Pagination fixes preserve visible text and explicit source page structure.

### Task 1: Shared OpenAPI contract and Swagger

**Files:** create `src/unflatten/service/api_openapi.py`, `api_docs.py` and local assets; modify `api.py`, `openapi_export.py`; test `tests/service/test_api_documentation.py` and schema tests.

**Interfaces:** shared `build_openapi(app)` supplies identical live/export schemas; documentation routes registered outside schema.

- [x] Write failing tests for public docs/schema, Bearer security, multipart file picker, headers and binary downloads.
- [x] Run tests and observe missing-doc/schema failures.
- [x] Add focused schema enrichment and local Swagger routes, regenerate committed OpenAPI and frontend types.
- [x] Verify API/auth/upload tests and frontend checks.

### Task 2: Headless instructions

**Files:** `README.md`, `README.sk.md`, `docs/API.md`, runnable configurable client example and focused client tests if needed.

**Interfaces:** existing `/v1/conversions`, polling and artifact routes from Task 1.

- [x] Document token retrieval/configuration without showing actual secrets, Swagger walkthrough, curl and request settings.
- [x] Demonstrate noninteractive conversion with configuration and an environment token.
- [x] Validate examples against test API and/or live local service.

### Task 3: Word pagination defect

**Files:** determined from root-cause investigation in renderer/service assembly; synthetic regression tests only.

- [x] Match screenshot to local output and compare IR, OOXML and rendered pages.
- [x] Record root cause and reproduce with a failing regression test.
- [x] Apply smallest assembly correction and verify renderer tests plus visual before/after.

### Task 4: Delivery

- [x] Run meaningful backend/frontend verification and independent code review.
- [x] Commit and push source changes to the repository; synchronize changed source into the local distribution.
- [x] Rebuild affected local services, preserving accounts, model and existing jobs.
- [x] Verify live Swagger upload/auth/status/Word download and provide links, corrected output and repository revision.
