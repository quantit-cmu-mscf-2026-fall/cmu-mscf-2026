"""Draft v3 claim specifications; exact quotations confer no semantic or financial proof.

All free-form hypothesis and mathematics fields are proposed research, never executable
code or established source facts. Only LocatedQuote entries are source statements.
ReviewedSourceInventory and qualified inputs must come from the trusted controller,
separately from the agent's claim payload. This module never assigns a budget or admits
a strategy; the controller retains the budget scope across aliases and revisions.

Claim v3 adds declared_defaults: researcher choices the source leaves open, recorded
with value and basis. Assessment v2 lists them in a non-blocking adaptation_record;
unresolved_rules still block. Saved v2 claims keep their identity and assess under
assessment v1 semantics.

Assessment v3 (now the judge of v3 claims) replaces "every reviewed span must be cited"
with a scope rule: every cited method/mathematics span must lie inside the review, and
at least one reviewed span must be cited that way. Saved assessment v2 records still load.

Claim v4 comes from structured required inputs: each is a canonical available name or
`unavailable:<free text>`, which no inventory qualifies, so it stays needs_data. Its
content and rules are otherwise v3's, so assessment v3 judges it; v3 claims are unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from capstone.factors.research_contracts import digest

CLAIM_VERSION = 4
ASSESSMENT_VERSION = 3
# claim semantics_version -> assessment version that judges it
ASSESSMENT_FOR_CLAIM = {2: 1, 3: 3, 4: 3}
# v2 judged v3 claims before the scope rule; its saved records stay loadable.
STORED_ASSESSMENT_VERSIONS = (1, 2, 3)
CLAIM_KINDS = ("reference", "adaptation", "exploratory_original")
QUOTE_PURPOSES = (
    "context",
    "mechanism",
    "prediction",
    "horizon",
    "universe",
    "falsifier",
    "mathematics",
    "required_inputs",
    "method",
)


def _text(value, name, *, blank=False):
    if not isinstance(value, str) or (not blank and not value.strip()):
        raise ValueError(f"{name} must be {'text' if blank else 'nonempty text'}")


def _identifier(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"invalid {name}")


def _strings(values, name):
    if not isinstance(values, (tuple, list)):
        raise ValueError(f"{name} must be a sequence")
    for value in values:
        _text(value, name)
    return tuple(sorted(set(values)))


def _records(values, kind, name):
    if not isinstance(values, (tuple, list)) or any(not isinstance(v, kind) for v in values):
        raise ValueError(f"{name} must contain only {kind.__name__} records")
    return tuple(sorted(set(values), key=lambda value: digest(asdict(value))))


@dataclass(frozen=True)
class LocatedQuote:
    """Exact substring of one immutable normalized extracted segment."""

    revision_id: str
    span_id: str
    quote: str
    start: int = 0
    purpose: str = "context"

    def __post_init__(self):
        _identifier(self.revision_id, "revision identity")
        _identifier(self.span_id, "span identity")
        _text(self.quote, "quote")
        if type(self.start) is not int or self.start < 0:
            raise ValueError("quote start must be a nonnegative integer")
        if self.purpose not in QUOTE_PURPOSES:
            raise ValueError("unknown quote attribution purpose")

    @property
    def end(self):
        return self.start + len(self.quote)


@dataclass(frozen=True)
class ResearchAssumption:
    text: str
    rationale: str

    def __post_init__(self):
        _text(self.text, "assumption")
        _text(self.rationale, "assumption rationale")


@dataclass(frozen=True)
class DeclaredDefault:
    """A researcher choice the source leaves open; labeled, never a source rule."""

    rule: str
    value: str
    basis: str

    def __post_init__(self):
        _text(self.rule, "declared default rule")
        _text(self.value, "declared default value")
        _text(self.basis, "declared default basis")


@dataclass(frozen=True)
class IdeaClaimV2:
    """A partial specification is retainable; assessment reports its unresolved fields."""

    kind: str
    mechanism: str = ""
    prediction: str = ""
    horizon: str = ""
    universe: str = ""
    falsifier: str = ""
    mathematics: str = ""
    method_key: str = ""
    required_inputs: tuple[str, ...] = ()
    source_statements: tuple[LocatedQuote, ...] = ()
    assumptions: tuple[ResearchAssumption, ...] = ()
    method_sources: tuple[str, ...] = ()
    unresolved_rules: tuple[str, ...] = ()
    adaptation_notes: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    semantics_version: int = CLAIM_VERSION
    declared_defaults: tuple[DeclaredDefault, ...] = ()

    def __post_init__(self):
        if self.kind not in CLAIM_KINDS or type(self.semantics_version) is not int:
            raise ValueError("unsupported claim kind or semantics version")
        if self.semantics_version not in ASSESSMENT_FOR_CLAIM:
            raise ValueError("unsupported claim semantics version")
        if self.semantics_version < 3 and self.declared_defaults:
            raise ValueError("declared defaults need claim semantics version 3")
        for name in (
            "mechanism",
            "prediction",
            "horizon",
            "universe",
            "falsifier",
            "mathematics",
            "method_key",
        ):
            _text(getattr(self, name), name, blank=True)
        for name in (
            "required_inputs",
            "method_sources",
            "unresolved_rules",
            "adaptation_notes",
            "aliases",
        ):
            object.__setattr__(self, name, _strings(getattr(self, name), name))
        for revision in self.method_sources:
            _identifier(revision, "method source identity")
        object.__setattr__(
            self,
            "source_statements",
            _records(self.source_statements, LocatedQuote, "source statements"),
        )
        object.__setattr__(
            self,
            "assumptions",
            _records(self.assumptions, ResearchAssumption, "researcher assumptions"),
        )
        object.__setattr__(
            self,
            "declared_defaults",
            _records(self.declared_defaults, DeclaredDefault, "declared defaults"),
        )

    @property
    def identity(self):
        semantics = asdict(self)
        semantics.pop("aliases")
        if self.semantics_version < 3:
            semantics.pop("declared_defaults")  # v2 identities predate the field
        return digest({"contract": "idea-claim", "semantics": semantics})

    @classmethod
    def from_dict(cls, payload):
        fields = dict(payload)
        fields["source_statements"] = tuple(
            LocatedQuote(**row) for row in fields.get("source_statements", ())
        )
        fields["assumptions"] = tuple(
            ResearchAssumption(**row) for row in fields.get("assumptions", ())
        )
        fields["declared_defaults"] = tuple(
            DeclaredDefault(**row) for row in fields.get("declared_defaults", ())
        )
        return cls(**fields)


@dataclass(frozen=True)
class MethodsReview:
    """A trusted review record, bound to its exact source revision and method spans.

    'sufficient' is the external reviewer's decision; archive extraction cannot make it.
    The inventory's provenance/authorization must be established by its trusted caller.
    """

    revision_id: str
    span_ids: tuple[str, ...]
    disposition: str
    reviewer: str
    review_record_id: str
    method_key: str

    def __post_init__(self):
        _identifier(self.revision_id, "review source identity")
        _text(self.reviewer, "reviewer")
        _text(self.review_record_id, "review record identity")
        _text(self.method_key, "reviewed method scope")
        object.__setattr__(self, "span_ids", _strings(self.span_ids, "reviewed spans"))
        for span_id in self.span_ids:
            _identifier(span_id, "reviewed span identity")
        if self.disposition not in ("sufficient", "insufficient", "unresolved"):
            raise ValueError("unsupported methods review disposition")
        if self.disposition == "sufficient" and not self.span_ids:
            raise ValueError("sufficient review requires located method spans")


@dataclass(frozen=True)
class ReviewedSourceInventory:
    authority_id: str
    reviews: tuple[MethodsReview, ...] = ()

    def __post_init__(self):
        _text(self.authority_id, "review inventory authority")
        reviews = _records(self.reviews, MethodsReview, "methods reviews")
        if len({(review.revision_id, review.method_key) for review in reviews}) != len(reviews):
            raise ValueError("ambiguous duplicate source reviews")
        object.__setattr__(self, "reviews", reviews)

    @property
    def identity(self):
        return digest({"contract": "reviewed-source-inventory-v1", **asdict(self)})


@dataclass(frozen=True)
class ClaimAssessment:
    """Independent axes; quote_integrity_verified establishes exact attribution only.

    A specification_ready result is not construction verification, scientific entailment,
    execution readiness or financial admission. Consumers must inspect all required axes.
    """

    claim_id: str
    budget_scope_id: str
    inventory_id: str
    input_inventory_id: str | None
    qualified_inputs: tuple[str, ...]
    evidence_status: str
    completeness_status: str
    data_status: str
    evidence_issues: tuple[str, ...]
    completeness_issues: tuple[str, ...]
    missing_inputs: tuple[str, ...]
    adaptation_record: tuple[str, ...] = ()
    assessment_version: int = ASSESSMENT_VERSION
    financial_status: str = field(default="pending", init=False)

    def __post_init__(self):
        if self.assessment_version not in STORED_ASSESSMENT_VERSIONS:
            raise ValueError("unsupported assessment version")
        object.__setattr__(
            self, "adaptation_record", _strings(self.adaptation_record, "adaptation record")
        )
        if self.assessment_version < 2 and self.adaptation_record:
            raise ValueError("adaptation records need assessment version 2")
        object.__setattr__(
            self, "qualified_inputs", _strings(self.qualified_inputs, "qualified inputs")
        )
        if self.input_inventory_id is not None:
            _text(self.input_inventory_id, "qualified input inventory identity")
        if (self.qualified_inputs or self.data_status == "ready") and not self.input_inventory_id:
            raise ValueError("qualified inputs require a trusted input inventory identity")
