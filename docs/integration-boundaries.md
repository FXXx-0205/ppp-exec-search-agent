# Integration Boundaries v1

This document defines the first production-facing integration seams for `AI Search Copilot`. The goal is to keep the workflow engine stable while allowing enterprise customers to connect their own systems incrementally.

## 1. Design Principles

- Keep workflow code independent from vendor-specific APIs.
- Separate read models, write models, and sync jobs.
- Preserve source attribution and timestamps for every imported record.
- Support partial availability: one connector failing should not take down the entire workflow.
- Treat every external system as untrusted input and normalize before downstream use.

## 2. CRM Boundary

Typical systems: Bullhorn, Salesforce, HubSpot, Invenias.

### Responsibilities

- Read client accounts, mandates, projects, and relationship history.
- Attach generated briefs and search notes back to the originating project.
- Enforce tenant and project scoping for all records.

### Suggested interface

- `list_projects(tenant_id, updated_since)`
- `get_project(project_id)`
- `get_client_account(account_id)`
- `append_project_note(project_id, markdown, metadata)`

### Normalized domain objects

- `ClientAccount`
- `SearchProject`
- `ProjectActivity`

## 3. ATS / Candidate System Boundary

Typical systems: Greenhouse, Lever, Ashby, internal candidate DB.

### Responsibilities

- Read candidate profiles, resumes, tags, status, and placement history.
- Write shortlist decisions and reviewer outcomes back when approved.
- Maintain immutable source references for evidence tracing.

### Suggested interface

- `search_candidates(filters, page_token)`
- `get_candidate(candidate_id)`
- `get_candidate_documents(candidate_id)`
- `upsert_shortlist_assessment(candidate_id, project_id, assessment)`

### Normalized domain objects

- `CandidateProfile`
- `CandidateDocument`
- `CandidateAssessment`

## 4. Document Store Boundary

Typical systems: SharePoint, Google Drive, Notion, Confluence, S3-backed knowledge stores.

### Responsibilities

- Read firm profiles, market maps, prior briefs, and search templates.
- Feed retrieval/indexing pipelines with metadata-rich documents.
- Track access policy and document provenance.

### Suggested interface

- `list_documents(collection, updated_since)`
- `get_document(document_id)`
- `get_document_content(document_id)`
- `store_generated_document(project_id, document_type, content, metadata)`

### Normalized domain objects

- `KnowledgeDocument`
- `DocumentChunk`
- `GeneratedArtifact`

## 5. Cross-Cutting Requirements

- Every connector must emit `source_system`, `source_id`, `synced_at`, `tenant_id`.
- Connector failures should map to typed integration errors, not generic 500s.
- Sync jobs should be idempotent and replayable.
- Secrets must be injected through environment or secret manager, never persisted in repo.
- Permissions should be evaluated before model invocation, not just before export.

## 6. Recommended Next Implementation Step

Create an `adapters/` layer with one protocol per boundary and keep workflow/services dependent only on those protocols. This lets us ship mock connectors first, then real enterprise connectors without rewriting ranking or briefing logic.
