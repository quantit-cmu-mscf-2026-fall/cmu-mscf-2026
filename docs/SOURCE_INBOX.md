# Source inbox: requests, evidence and the batch command

The source inbox archives supplied evidence (a URL, a local PDF or text file, or
inline text) as immutable, located text. It makes no model call, reads no licensed
data, and does not touch the holdout. It does not verify a paper's methods,
qualify market data or produce a strategy: "ready" means readable evidence was
archived, nothing more.

Install the optional PDF extractor with `pip install -e ".[intake]"` (pypdf 6.19.0).
Without it a PDF is blocked as `needs_pdf_dependency`; HTML and text still work.
HTTP uses `requests`, already a core dependency. Transport tests use fakes, never
live websites.

## Requests

`capstone.factors.source_intake.load_source_requests(path)` reads a JSON or TOML
table with a `sources` list, or a text file with one `http(s)` URL per non-comment
line. At most 1,000 sources and 1 MB; unknown fields are refused; local paths are
relative to the manifest. A row may carry `kind` (`url`, `local_pdf`, `local_text`,
`inline_text`), `key`, `title`, `author`, `published_date` and `role` (`seed`,
`reference`, `adaptation`). A TOML example:

```toml
[[sources]]
kind = "url"
locator = "https://arxiv.org/abs/2101.00001"
role = "seed"

[[sources]]
kind = "local_pdf"
locator = "paper.pdf"
key = "my-paper"

[[sources]]
kind = "inline_text"
key = "original-attention-idea"
title = "An observation to investigate"
text = "Perhaps unusually high attention precedes a reversal. This is an untested idea."
role = "seed"
```

Title, author and published date are attribution you supply, not extracted or
independently verified bibliography.

## Identities

- `source_id` / `work_id`: the explicit `key`, else the kind and locator (or inline
  text). A work keeps its identity across revisions when you supply a `key`.
- `request_id`: a digest of every supplied field.
- `EvidenceSpan`: exact text, its locator and offsets, and a SHA-256 of the text.
- `SourceArtifact.revision_id`: a digest of one acquired revision.

Distinct requests and revisions remain distinct evidence. These identities create no
statistical families or trial budgets.

## Acquisition

- Local files and inline text are read with a byte bound (default 5 MB). A local
  kind never accepts an `http`/`file` locator.
- URLs are fetched only from open-access hosts: arXiv, SSRN and NBER (and their
  subdomains). Any other host, and any redirect off these hosts, is refused as
  `unsupported_url` before a request is made. A URL with credentials or
  authentication query parameters is refused too.
- HTTP is public-address only: DNS answers and the connected peer are checked
  before any request data is sent, redirects are followed manually and validated at
  every hop (at most 3), and the session takes no cookies, netrc or proxy
  credentials from the machine. Denials (401/403/429), size limits and unsupported
  media are recorded with a named reason and the bytes retained.
- A page that blocks access is not worked around. Paste the text as a separate
  `inline_text` request, with the original URL as its title or key, and it remains
  an excerpt, not a claim that the whole source was read.

## Extraction

Original bytes, exact extracted segments, the extraction version and locators are
retained. HTML whitespace is normalized and scripts, styles and navigation are
excluded (article, then main, then body); PDF extraction is page located. Neither
implies a formula, table or method was understood. Encrypted, scanned or unreadable
documents get an explicit blocker, and a PDF where some pages yield text and others
none is blocked as `pdf_partial_requires_ocr`: its bytes are kept but no spans, so a
page subset is never presented as the source. pypdf is not an OCR engine.

## Run

```text
python -m capstone.pipeline.source_batch ingest --sources sources.toml --out sources_out/first
python -m capstone.pipeline.source_batch status --out sources_out/first
python -m capstone.pipeline.source_batch resume --out sources_out/first
```

`ingest` freezes the canonical requests (row order and duplicates do not matter),
the byte limit and an engine identity (hashes of the intake code, the Python
version and the pypdf/requests/urllib3 versions) before any acquisition.
`SOURCE_INBOX.html` in the output directory is the readable result. Exit code 0
means every unique source was extracted; 2 means at least one is blocked or in
error (the others are still retained). Neither means a strategy was generated or
validated. Keep source content and output directories out of Git.

## Resume, changes and corruption

Resume verifies every cached revision, including superseded ones, and the summary's
checkpoint (journal sequence and event id) before acquiring anything, and never
silently retries even a blocked source. `resume --refresh` explicitly reacquires
requests: changed bytes or extraction versions create new revisions, and existing
revisions are never overwritten. Changed inputs, byte limit or engine require a new
batch; `status` can still inspect an older batch without its original extractor and
never writes. Corruption is an error, not permission to fetch a replacement; a
damaged or truncated journal is preserved and refuses further appends.

Archive and batch use process-owned locks, so concurrent writers fail instead of
interleaving. Objects and revisions are published atomically, and each outcome is
journaled in an append-only hash chain. A crash may leave complete events beyond the
last summary; they are retained.

## Offline verification

```text
python -m pytest -q tests/test_source_intake.py tests/test_source_extract.py tests/test_source_acquisition.py tests/test_source_archive.py tests/test_source_batch.py
```

These establish extraction, bounds, refusals, immutable revisions, corruption and
fresh-process resume on synthetic inputs. They do not establish financial
performance, model quality or the content of any live source.
