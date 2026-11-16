# Source inbox: requests and evidence

A source request names one piece of supplied evidence (a URL, a local PDF or text
file, or inline text) without fetching it. `capstone.factors.source_intake`
holds the immutable contracts the later acquisition, extraction and archive steps
share. Nothing here calls a model, reads licensed data or touches the holdout.

## Manifest

`load_source_requests(path)` reads a JSON or TOML table with a `sources` list, or a
text file with one `http(s)` URL per non-comment line. At most 1,000 sources and
1 MB. Unknown fields are refused. Local paths are relative to the manifest. Each row
may carry `kind` (`url`, `local_pdf`, `local_text`, `inline_text`), `key`, `title`,
`author`, `published_date` and `role` (`seed`, `reference`, `adaptation`).

## Identities

- `source_id` / `work_id`: the explicit `key`, else the kind and locator (or inline
  text). A work keeps its identity across revisions when you supply a `key`.
- `request_id`: a digest of every supplied field.
- `EvidenceSpan`: exact text, its locator and offsets, and a SHA-256 of the text.
- `SourceArtifact.revision_id`: a digest of one acquired revision.

URLs with embedded credentials or authentication query parameters are refused
before anything is stored. Keep source content and outputs private and out of Git.
