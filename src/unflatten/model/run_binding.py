"""RunManifestBindingVerifier: RFC005 S1 Run Manifest verification (design
§8 "Privacy admission binding", §16 criteria 12 and 14).

Closes the privacy fail-open governance found in round two: hash equality
between a cache and a lock proves the two *agree*, not that what they agree
on is admissible -- a `dev_only` cache satisfies a `dev_only` lock just as
well as a `strict` one satisfies a `strict` one. `s1_run_manifest` pins
`required_admission` and `gradient_bearing` as explicit literal fields
inside its own frozen, hash-bound payload, and this module is the one place
that reads and constrains those values. A `required_admission` supplied by
a CLI flag, an environment variable, or any other runtime channel is not a
pin and this module never reads one from anywhere but the frozen manifest
dict it is handed.

Name collision, stated first because it is the likeliest way to get this
wrong: `run_manifest.json` already exists in this repository as an
unrelated dataset-factory artifact
(`src/unflatten/dataset/synthetic/factory.py:328,348`, read by
`src/unflatten/dataset/split/release.py:144-151,444-449` and
`src/unflatten/dataset/gates/engine.py:386-436`). It carries
`documents`/`recipe_id`, not training bindings. The artifact this module
verifies is `s1_run_manifest`
(`schemas/s1_run_manifest.schema.json`) -- a different thing, and nothing
here reads or writes the dataset-side file.

`gradient_bearing` is read from the manifest, never derived from the
adjacent `run_class` (`discovery | promotion`): neither value encodes an
optimizer step, and the first S1 run is expected to be `discovery` while
being gradient-bearing (design §8 rule 3a).

M1 ships this verifier, not a populated manifest and not a run: it is pure
verification over plain dicts. No model, no gradient, no run execution, no
parameterized layer, no optimizer (design §16 criterion 15).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, fields
from functools import cache
from pathlib import Path
from typing import Any

import jsonschema.protocols
import jsonschema.validators

from unflatten.model.capability_profile import (
    TYPED_PROPERTIES_VALUES,
    CapabilityProfile,
)
from unflatten.model.composite_vocab import qwen_text_allowlist_hash
from unflatten.model.exact_text_cache import (
    Admission,
    CacheExpectation,
    ExactTextCacheIdentity,
    ExactTextCacheIncompatible,
)
from unflatten.model.hashing import domain_hash
from unflatten.model.manifests import component_lock_hash
from unflatten.model.stage_policy import StagePolicyInvalid, stage_policy_hash

_SCHEMA_DIR = Path(__file__).parent / "schemas"

#: `s1_run_manifest_hash = domain_hash(_RUN_MANIFEST_DOMAIN, payload_without_self_hash)`
#: (design §8). Owned here: no other module computes an `s1_run_manifest_hash`.
_RUN_MANIFEST_DOMAIN = "rfc005.s1-run-manifest/v1"

#: Hash binding an `s1_run_manifest` to the full `ExactTextCacheIdentity`
#: payload it was frozen against. Owned here: `exact_text_cache.py`
#: stores the identity payload itself (in its manifest's `identity` key)
#: but does not hash it -- that hash exists only for this binding, so this
#: module is its sole owner rather than a second copy of a domain string.
_CACHE_IDENTITY_DOMAIN = "rfc005.exact-text-cache-identity/v1"

#: Hash binding an `s1_run_manifest` to the frozen cohort expectation
#: (`CacheExpectation`, DEC-MP-4) the run must read its cache under. Owned
#: here for the same reason as `_CACHE_IDENTITY_DOMAIN`: the payload lives
#: in `exact_text_cache.py`, the BINDING is a run-manifest concern.
_CACHE_EXPECTATION_DOMAIN = "rfc005.exact-text-cache-expectation/v1"

#: `required_admission` reuses the same closed vocabulary as
#: `ExactTextCacheIdentity.admission` -- one type, not a second Literal
#: that could drift from it.
type RequiredAdmission = Admission

#: The direct hash bindings carried on both `s1_run_manifest` and
#: `ExactTextCacheIdentity` (design §8 table): checked field-for-field so a
#: mutation of either payload is caught even though `component_lock_hash`
#: also participates in `exact_text_cache_identity_hash`.
_DIRECT_BINDING_FIELDS: tuple[str, ...] = (
    "component_lock_hash",
    "training_target_materializer_hash",
    "loss_unit_schema_hash",
    "stage_policy_hash",
    "conditioning_policy_hash",
    "runtime_limits_policy_hash",
    "head_registry_hash",
)


@cache
def _validator() -> jsonschema.protocols.Validator:
    with open(_SCHEMA_DIR / "s1_run_manifest.schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def compute_s1_run_manifest_hash(payload_without_self_hash: dict[str, Any]) -> str:
    """`s1_run_manifest_hash` for a payload with its own
    `s1_run_manifest_hash` field already removed. Public so a caller
    freezing a manifest computes the id the same way this module verifies
    it, without a second copy of `_RUN_MANIFEST_DOMAIN`."""
    return domain_hash(_RUN_MANIFEST_DOMAIN, payload_without_self_hash)


def compute_exact_text_cache_identity_hash(identity_payload: dict[str, Any]) -> str:
    """Hash binding an `s1_run_manifest` to one `ExactTextCacheIdentity`
    payload. Public for the same reason as
    :func:`compute_s1_run_manifest_hash`: whoever freezes a manifest must
    compute this the same way this module verifies it."""
    return domain_hash(_CACHE_IDENTITY_DOMAIN, identity_payload)


def compute_exact_text_cache_expectation_hash(
    expectation_payload: dict[str, Any],
) -> str:
    """Hash binding an `s1_run_manifest` to one frozen cohort expectation
    payload (`CacheExpectation.payload()`). Public for the same reason as
    :func:`compute_exact_text_cache_identity_hash`."""
    return domain_hash(_CACHE_EXPECTATION_DOMAIN, expectation_payload)


class RunManifestBindingVerifier:
    """Verifies a frozen `s1_run_manifest` payload against one
    `ExactTextCacheIdentity` payload (as a plain dict, e.g.
    `dataclasses.asdict(identity)`), refusing the run on any omission,
    mutation, or inadmissible privacy binding (design §8, §16 criteria 12
    and 14).

    Every refusal raises the M1-internal `ExactTextCacheIncompatible`
    (`exact_text_cache.py`, §9.3) -- never the unratified
    `EXACT_TEXT_CACHE_INCOMPATIBLE` Issue code, which would itself breach
    the RR-3 boundary (§16 criterion 17).

    `allowlist` is the `qwen_text_allowlist` payload bound to this run
    (design §4.2 "How enforcement must work", variant E1). It is optional
    only for a non-gradient-bearing run: `gradient_bearing = true` requires
    it, and its absence is a refusal rather than a silently skipped check.
    `control_text_check` is never promoted into `ExactTextCacheIdentity` or
    `s1_run_manifest` -- that would copy a claim the allowlist artifact
    already makes into a second place that could disagree with it.

    `capability_profile` is the `model_capability_profile` this run is
    bound to, presented for the same reason and on the same terms as
    `allowlist`: required for a gradient-bearing run, refused unless it is
    the profile the cache identity names. See
    `_verify_typed_property_profile` for what it does and does not prove.
    """

    def verify(
        self,
        run_manifest: dict[str, Any],
        cache_identity: dict[str, Any],
        allowlist: dict[str, Any] | None = None,
        capability_profile: CapabilityProfile | None = None,
        expectation: CacheExpectation | None = None,
    ) -> None:
        self._validate_schema(run_manifest)
        self._verify_self_hash(run_manifest)
        self._verify_admission(run_manifest, cache_identity)
        self._verify_bindings(run_manifest, cache_identity)
        self._verify_allowlist_is_bound(cache_identity, allowlist)
        self._verify_control_text_check(run_manifest, allowlist)
        self._verify_typed_property_profile(run_manifest, cache_identity, capability_profile)
        self._verify_expectation_is_bound(run_manifest, expectation)
        self._verify_run_config(run_manifest, expectation)

    def _validate_schema(self, run_manifest: dict[str, Any]) -> None:
        errors = sorted(
            _validator().iter_errors(run_manifest), key=lambda e: list(e.absolute_path)
        )
        if errors:
            first = errors[0]
            pointer = "/" + "/".join(str(p) for p in first.absolute_path)
            raise ExactTextCacheIncompatible(
                f"s1_run_manifest invalid at {pointer or '/'}: {first.message} "
                f"-- an absent, null, or malformed binding is refused, never "
                f"defaulted or coerced"
            )

    def _verify_self_hash(self, run_manifest: dict[str, Any]) -> None:
        payload = {k: v for k, v in run_manifest.items() if k != "s1_run_manifest_hash"}
        expected = compute_s1_run_manifest_hash(payload)
        if run_manifest["s1_run_manifest_hash"] != expected:
            raise ExactTextCacheIncompatible(
                "s1_run_manifest_hash does not match its own payload -- the "
                "manifest was mutated after freezing, or its hash was never "
                "computed correctly"
            )

    def _verify_admission(
        self, run_manifest: dict[str, Any], cache_identity: dict[str, Any]
    ) -> None:
        gradient_bearing = run_manifest["gradient_bearing"]
        required_admission = run_manifest["required_admission"]

        if gradient_bearing and required_admission != "strict":
            raise ExactTextCacheIncompatible(
                f"gradient_bearing=True requires required_admission='strict', "
                f"got {required_admission!r} -- refused (design §8 rule 3a, "
                f"§16 criterion 14)"
            )

        cache_admission = cache_identity.get("admission")
        if cache_admission != required_admission:
            raise ExactTextCacheIncompatible(
                f"cache admission {cache_admission!r} != run_manifest "
                f"required_admission {required_admission!r} -- hash equality "
                f"elsewhere does not make an inadmissible cache admissible "
                f"(design §8 rule 4)"
            )

        if gradient_bearing and run_manifest["head_registry_hash"] is None:
            raise ExactTextCacheIncompatible(
                "gradient_bearing=True requires a non-null head_registry_hash; "
                "a cache built without the seventh `--bindings` key (pre-2.0.0) "
                "carries head_registry_hash=null and is therefore not "
                "gradient-admissible (design §8 rule 3, ERRATUM E1)"
            )

    def _verify_bindings(
        self, run_manifest: dict[str, Any], cache_identity: dict[str, Any]
    ) -> None:
        missing = [
            f.name for f in fields(ExactTextCacheIdentity) if cache_identity.get(f.name) is None
        ]
        if missing:
            raise ExactTextCacheIncompatible(
                "cache_identity is missing (or null on) required field(s): "
                + ", ".join(sorted(missing))
                + " -- an absent/null identity field is refused, never defaulted"
            )

        expected_identity_hash = compute_exact_text_cache_identity_hash(cache_identity)
        if run_manifest["exact_text_cache_identity_hash"] != expected_identity_hash:
            raise ExactTextCacheIncompatible(
                "exact_text_cache_identity_hash does not match the supplied "
                "cache identity -- this catches omission or mutation of any "
                "tokenizer, allowlist, vocabulary, grammar, profile, or cache "
                "binding transitively (design §16 criterion 12)"
            )

        for field_name in _DIRECT_BINDING_FIELDS:
            manifest_value = run_manifest[field_name]
            identity_value = cache_identity.get(field_name)
            if manifest_value != identity_value:
                raise ExactTextCacheIncompatible(
                    f"{field_name} mismatch: s1_run_manifest has "
                    f"{manifest_value!r}, cache identity has "
                    f"{identity_value!r} -- refused (design §16 criterion 12)"
                )

    def _verify_allowlist_is_bound(
        self,
        cache_identity: dict[str, Any],
        allowlist: dict[str, Any] | None,
    ) -> None:
        """The supplied `allowlist` payload must be the one the cache
        identity names. `_verify_bindings` hashes the identity *dict*, in
        which `qwen_text_allowlist_hash` is only a string field -- that
        proves the identity is intact, not that the payload handed to
        `verify()` is the payload it names.

        Deliberately outside `_verify_control_text_check` and outside any
        `gradient_bearing` branch: requiring `control_text_check ==
        'applied'` is legitimately gradient-only, but "is this the payload
        you bound?" is not -- a diagnostic run must not be verifiable
        against an allowlist it never bound either. Deliberately after
        `_verify_bindings`, so a null/absent identity field is reported as
        the missing binding it is rather than as a hash mismatch.
        """
        if allowlist is None:
            # Absence is `_verify_control_text_check`'s call: required for a
            # gradient-bearing run, legitimate for a diagnostic one.
            return

        supplied = qwen_text_allowlist_hash(allowlist)
        bound = cache_identity["qwen_text_allowlist_hash"]
        if supplied != bound:
            raise ExactTextCacheIncompatible(
                f"the supplied qwen_text_allowlist payload is not the payload "
                f"this run is bound to: it hashes to {supplied!r}, the cache "
                f"identity binds {bound!r} -- what the payload claims about "
                f"itself (including control_text_check) says nothing until it "
                f"is the bound one (design §4.8)"
            )

    def _verify_control_text_check(
        self,
        run_manifest: dict[str, Any],
        allowlist: dict[str, Any] | None,
    ) -> None:
        gradient_bearing = run_manifest["gradient_bearing"]
        if not gradient_bearing:
            return

        if allowlist is None:
            raise ExactTextCacheIncompatible(
                "gradient_bearing=True requires the qwen_text_allowlist payload "
                "(the `allowlist` argument to verify()); its absence is a "
                "refusal, not a skipped check (design §4.2 variant E1)"
            )
        control_text_check = allowlist.get("control_text_check")
        if control_text_check != "applied":
            raise ExactTextCacheIncompatible(
                f"gradient_bearing=True requires the allowlist's "
                f"control_text_check == 'applied', got {control_text_check!r} "
                f"-- a 'skipped' allowlist never decoded candidate ids for "
                f"control-bearing text and cannot back a gradient-bearing run "
                f"(design §4.2)"
            )

    def _verify_typed_property_profile(
        self,
        run_manifest: dict[str, Any],
        cache_identity: dict[str, Any],
        capability_profile: CapabilityProfile | None,
    ) -> None:
        """The PROFILE half of DEC-017/DEC-021 reading **B**: a
        gradient-bearing run bound to a non-null `head_registry_hash` must
        be bound to a capability profile that ADMIITS typed properties,
        i.e. one whose `typed_properties` is the closed-domain value
        `"present_frozen"` (`TYPED_PROPERTIES_VALUES`). The decision is a
        closed allowlist, not a denylist of the `"not_applicable"` literal
        (R3, `ZOSTAVA-typovany-head-modul.md`): a profile whose
        `typed_properties` is outside the domain is refused as malformed
        before the admit/deny decision is even made.

        WHAT THIS DOES NOT COVER, and a future reader must not read a
        green gate as more than it is: reading B says a non-null
        `head_registry_hash` means the typed heads really govern the run.
        Its other half -- that some `typed_heads.*` group is actually
        TRAINABLE in `trainable_group_registry` -- is NOT checked here and
        arrives at S3 (design §9 O3, owner 2026-09-09: option (ii) now,
        option (i) at S3). Enforcing the trainable half today would refuse
        the very cache this milestone builds, whose typed heads are frozen
        by construction. So a run passing this check has proved that its
        cache was built under a profile that admits typed properties --
        not that any gradient will ever reach a typed head.

        Why the profile is PRESENTED by the caller rather than looked up
        here: neither `s1_run_manifest` nor `ExactTextCacheIdentity`
        carries `typed_properties` -- the identity carries only
        `capability_profile_hash`, and no artifact on disk carries the
        profile payload. A gate that resolved that hash against an
        in-repo constant would answer the question from the same place
        the build answered it: self-agreement, not verification. So the
        caller presents the profile and it is refused unless it is the
        one the cache identity names, exactly as
        `_verify_allowlist_is_bound` treats the allowlist. Presenting an
        in-repo constant is fine and is what the callers do: a cache
        built under a `not_applicable` profile is refused whichever
        profile is presented -- the right one on `typed_properties`, any
        other on the binding. Absence is a refusal for a gradient-bearing
        run, never a skipped check.

        The argument is a `CapabilityProfile`, not a raw payload, because
        the binding is checked against `profile_hash`, which
        `build_capability_profile` computes from the validated payload;
        a free dict would let a caller assert a hash it never derived.
        """
        if not run_manifest["gradient_bearing"]:
            return
        if run_manifest["head_registry_hash"] is None:
            # `_verify_admission` already refused this pair; the conjunct
            # this method enforces is not even reached.
            return

        if capability_profile is None:
            raise ExactTextCacheIncompatible(
                "gradient_bearing=True with a non-null head_registry_hash "
                "requires the bound model_capability_profile (the "
                "`capability_profile` argument to verify()); its absence is a "
                "refusal, not a skipped check -- a non-null head registry hash "
                "must mean the typed heads govern this run (DEC-017 reading B, "
                "profile half)"
            )

        supplied = capability_profile.profile_hash
        bound = cache_identity["capability_profile_hash"]
        if supplied != bound:
            raise ExactTextCacheIncompatible(
                f"the supplied model_capability_profile is not the capability "
                f"profile this run is bound to: it hashes to {supplied!r}, the "
                f"cache identity binds {bound!r} -- what a profile claims "
                f"about its own typed_properties says nothing until it is the "
                f"bound one"
            )

        typed_properties = capability_profile.typed_properties
        if typed_properties not in TYPED_PROPERTIES_VALUES:
            raise ExactTextCacheIncompatible(
                f"bound capability profile declares "
                f"typed_properties={typed_properties!r}, which is outside the "
                f"closed domain {sorted(TYPED_PROPERTIES_VALUES)} -- a "
                f"CapabilityProfile with an out-of-domain value is refused "
                f"before the typed-properties admission decision "
                f"(R3, ZOSTAVA-typovany-head-modul.md)"
            )
        if typed_properties != "present_frozen":
            raise ExactTextCacheIncompatible(
                f"gradient_bearing=True with a non-null head_registry_hash "
                f"requires the bound capability profile to admit typed "
                f"properties, i.e. typed_properties='present_frozen', got "
                f"typed_properties={typed_properties!r} -- a cache built "
                f"under a profile that admits no typed property cannot back "
                f"a run whose head registry hash claims typed heads govern "
                f"it (DEC-017 reading B, profile half; the trainable half "
                f"lands at S3)"
            )

    def _verify_expectation_is_bound(
        self,
        run_manifest: dict[str, Any],
        expectation: CacheExpectation | None,
    ) -> None:
        """DEC-MP-4: the cohort a gradient-bearing run reads is pinned in
        this frozen manifest, and the payload the run actually uses must be
        the pinned one.

        Presented rather than looked up, exactly as
        `_verify_allowlist_is_bound` treats the allowlist: resolving the
        hash against whatever file the run happened to load would answer
        the question from the same place the run answered it.

        Absence is a refusal for a gradient-bearing run, never a skipped
        check -- an unpinned cohort is the state in which a partial cache
        trains as though it were a full one, which is the entire defect
        this binding closes. A non-gradient-bearing run may verify without
        one (a diagnostic read is allowed to look at whatever cache it is
        pointed at), but if it presents an expectation it is still held to
        the pinned hash.
        """
        pinned = run_manifest["exact_text_cache_expectation_hash"]
        gradient_bearing = run_manifest["gradient_bearing"]

        if gradient_bearing and pinned is None:
            raise ExactTextCacheIncompatible(
                "gradient_bearing=True requires a non-null "
                "exact_text_cache_expectation_hash: the expected page key set E "
                "and approved exclusions X must be pinned in the frozen "
                "manifest (DEC-MP-4), because a cache's own scope declaration "
                "agrees with itself whatever it contains"
            )
        if expectation is None:
            if gradient_bearing:
                raise ExactTextCacheIncompatible(
                    "gradient_bearing=True requires the frozen cohort "
                    "expectation payload (the `expectation` argument to "
                    "verify()); its absence is a refusal, not a skipped check "
                    "(DEC-MP-4)"
                )
            return

        supplied = compute_exact_text_cache_expectation_hash(expectation.payload())
        if supplied != pinned:
            raise ExactTextCacheIncompatible(
                f"the supplied cohort expectation is not the one this run is "
                f"bound to: it hashes to {supplied!r}, the manifest pins "
                f"{pinned!r} -- an expectation the run chose at runtime is not "
                f"a pin (DEC-MP-4)"
            )

    def _verify_run_config(
        self, run_manifest: dict[str, Any], expectation: CacheExpectation | None
    ) -> None:
        """1.2.0 (owner decision 2026-09-13, #19 B / #21 C): a run that
        spends optimizer steps must carry the budget it may spend -- and
        the budget may not exceed the cohort.

        Two refusals, both gradient-only:

        * a gradient-bearing manifest whose `run_config` is null: a
          diagnostic run spends no budget, a gradient-bearing one must
          freeze its steps, sampler, augmentation and optimizer in the
          manifest rather than inheriting CLI defaults;
        * a budget whose `steps x sampler.batch_size` exceeds the pinned
          cohort's page count: the S1 canary trains at most one epoch over
          (bundle, page), and the cohort is the pinned expectation, never
          the cache's own scope declaration.
        """
        if not run_manifest["gradient_bearing"]:
            return
        run_config = run_manifest["run_config"]
        if run_config is None:
            raise ExactTextCacheIncompatible(
                "gradient_bearing=True requires a non-null run_config: a run "
                "that spends optimizer steps must freeze its budget (steps, "
                "sampler, augmentation, optimizer) in the manifest -- a null "
                "run_config is a refusal, never an unfrozen run (1.2.0, #21 C)"
            )
        if expectation is None:
            return
        # #19 B scopes the one-epoch cap to the S1 canary. Later programs
        # bind through this same manifest but bound their exposure by their
        # own frozen plan -- DEC-041's production run is three page-uniform
        # epochs, each page exactly three times (s4_plan, verified at
        # admission). Found when the v15 freeze refused 236 427 steps.
        if run_manifest["program_id"] != S1_CANARY_PROGRAM_ID:
            return
        cohort = len(expectation.target_keys())
        exposure = run_config["steps"] * run_config["sampler"]["batch_size"]
        if exposure > cohort:
            raise ExactTextCacheIncompatible(
                f"run_config exposure {run_config['steps']} steps x "
                f"{run_config['sampler']['batch_size']} batch = {exposure} "
                f"exceeds the pinned cohort's {cohort} page(s) -- the S1 "
                "canary trains at most one epoch over (bundle, page), never "
                "more (1.2.0, #19 B)"
            )


# ---------------------------------------------------------------------------
# Producer (design §8)
# ---------------------------------------------------------------------------

#: The one program #19 B's one-epoch exposure cap binds (owner decision
#: 2026-09-13: "S1 canary = fixný single-cell cohort ... ≤ 1 epoch").
S1_CANARY_PROGRAM_ID = "unflatten-training-v0/s1-canary@1"

#: The `s1_run_manifest` schema revision this producer emits.
#: 1.2.0: `program_id`, `stage_id` (derived from the bound stage policy
#: document, never typed) and `run_config` (the run's own budget) -- owner
#: decision 2026-09-13, #17 C / #21 C. 1.1.0:
#: `exact_text_cache_expectation_hash` (DEC-MP-4), required by the schema
#: and non-null for a gradient-bearing run. A 1.0.0 manifest omits the
#: keys entirely and fails schema validation rather than being read as an
#: unattributed, unpinned run.
S1_RUN_MANIFEST_SCHEMA_VERSION = "1.2.0"

#: Bindings a manifest reads off the cache identity it is frozen against,
#: rather than accepting separately. Two arguments for one binding is two
#: places that can disagree, and `_verify_bindings` would then refuse the
#: manifest its own builder produced.
_DERIVED_FROM_IDENTITY: tuple[str, ...] = (
    "training_target_materializer_hash",
    "loss_unit_schema_hash",
    "stage_policy_hash",
    "conditioning_policy_hash",
    "runtime_limits_policy_hash",
    "head_registry_hash",
)


def build_s1_run_manifest(
    *,
    run_id: str,
    run_class: str,
    gradient_bearing: bool,
    required_admission: RequiredAdmission,
    component_lock: dict[str, Any],
    cache_identity: ExactTextCacheIdentity,
    program_id: str,
    stage_policy: Mapping[str, Any],
    allowlist: dict[str, Any] | None = None,
    capability_profile: CapabilityProfile | None = None,
    expectation: CacheExpectation | None = None,
    run_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze one `s1_run_manifest` for `cache_identity`, verified on the
    way out by :class:`RunManifestBindingVerifier` itself.

    Every hash binding is derived: `component_lock_hash` is recomputed from
    the real `component_lock` document (and refused if the cache identity
    names a different one -- the check RFC005 §14.4 gate 14 was missing),
    `exact_text_cache_identity_hash` from the identity, and the six
    bindings in :data:`_DERIVED_FROM_IDENTITY` are read off the identity so
    the manifest cannot name a materializer, LossUnit schema, stage
    policy, conditioning policy, limits policy or head registry the cache
    was not written under. `stage_id` is the seventh derivation (1.2.0,
    #17 C): read off the `stage_policy` document the caller presents, which
    is refused unless it hashes to the identity's `stage_policy_hash` -- a
    manifest can never name a stage its bound stage policy does not.

    The five fields that are NOT derived -- `run_id`, `run_class`,
    `gradient_bearing`, `required_admission`, `program_id` -- are
    declarations about the run, which is what this artifact exists to
    freeze; none has a default and `gradient_bearing` is never inferred
    from `run_class` (design §8 rule 3a). `exact_text_cache_expectation_hash`
    and `run_config` are the same kind of declaration (DEC-MP-4; 1.2.0
    #19 B / #21 C): the cohort and the run's own budget are pinned from the
    caller's frozen values, never read off the cache, which has no
    independent knowledge of what it should contain or how much it may
    spend.

    Building a manifest is not authorization to run: a `gradient_bearing=
    True` manifest with a null head registry (or a non-`strict` admission,
    or a `capability_profile` that is absent, unbound, or declares
    `typed_properties = "not_applicable"`, or a null `run_config`, or a
    `run_config` that spends more than one epoch over the pinned cohort) is
    refused here exactly as the verifier refuses it at run time.
    """
    identity_payload = asdict(cache_identity)

    lock_hash = component_lock_hash(component_lock)
    if lock_hash != identity_payload["component_lock_hash"]:
        raise ExactTextCacheIncompatible(
            f"the supplied component_lock hashes to {lock_hash!r}, but the cache "
            f"identity binds component_lock_hash "
            f"{identity_payload['component_lock_hash']!r} -- the cache was written "
            "under a different lock (RFC005 §14.4 gate 14)"
        )

    try:
        supplied_stage_policy_hash = stage_policy_hash(stage_policy)
    except StagePolicyInvalid as exc:
        raise ExactTextCacheIncompatible(
            f"the supplied stage policy document is refused by the stage "
            f"policy validator: {exc}"
        ) from exc
    if supplied_stage_policy_hash != identity_payload["stage_policy_hash"]:
        raise ExactTextCacheIncompatible(
            "the supplied stage policy document is not the one the cache "
            f"identity binds: it hashes to {supplied_stage_policy_hash!r}, "
            f"the identity binds {identity_payload['stage_policy_hash']!r} "
            "-- stage_id and the loss policy are derived from the bound "
            "document, never typed (1.2.0, #17 C)"
        )

    payload: dict[str, Any] = {
        "s1_run_manifest_schema_version": S1_RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "run_class": run_class,
        "gradient_bearing": gradient_bearing,
        "required_admission": required_admission,
        "component_lock_hash": lock_hash,
        "exact_text_cache_identity_hash": compute_exact_text_cache_identity_hash(
            identity_payload
        ),
        "exact_text_cache_expectation_hash": (
            None
            if expectation is None
            else compute_exact_text_cache_expectation_hash(expectation.payload())
        ),
        "program_id": program_id,
        "stage_id": stage_policy["stage_id"],
        "run_config": dict(run_config) if run_config is not None else None,
        **{name: identity_payload[name] for name in _DERIVED_FROM_IDENTITY},
    }
    manifest = dict(payload)
    manifest["s1_run_manifest_hash"] = compute_s1_run_manifest_hash(payload)

    # The producer's output is checked by the same verifier the run uses:
    # a manifest this module could build but that module would refuse
    # would be worse than no producer.
    RunManifestBindingVerifier().verify(
        manifest, identity_payload, allowlist, capability_profile, expectation
    )
    return manifest
