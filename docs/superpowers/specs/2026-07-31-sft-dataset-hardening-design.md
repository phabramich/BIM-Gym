# SFT Dataset Hardening Design

## Goal

Make BIM-Gym SFT examples information-complete, independently reproducible,
structurally diverse, and difficult for shortcut heuristics while retaining a
deterministic executable verifier.

## Scope

This change hardens the existing phase-0 dataset rather than introducing a new
training protocol. It covers five connected areas:

1. the information shown to the model;
2. the serialized dataset record and provenance;
3. generator diversity and task difficulty;
4. semantic validation and dataset diagnostics;
5. held-out split independence.

The existing action and predicate DSL remains the execution contract. Reference
actions remain witnesses of solvability, not the only accepted solutions.

## Model-visible task contract

Every user message will contain a canonical JSON task context after the natural
language instruction. The context contains:

- a schema version;
- every scene entity needed to solve or disambiguate the task, including its ID,
  type, parent, placement, bbox and relevant hosting properties;
- the complete formal goal predicate list;
- units (`mm`) and coordinate convention.

Entities and object keys are deterministically ordered. Numeric values use the
same precision as the runtime scene. The context is derived from the actual
`TaskSpec`/`Scene`, never independently reconstructed by a template.

Queries may still be included to teach the action vocabulary, but a one-shot SFT
example must not rely on a query result that the model cannot observe before
generating a later action. Therefore all facts required by a dependent action
must already be present in the canonical context. Genuine query/error recovery
will be represented only when the record format includes an intervening tool
observation; until that protocol exists, fabricated zero-move recovery examples
are disabled.

## Self-contained records and provenance

`SftMeta` will include:

- `generator_version`;
- `scene_seed`;
- `task_spec`, serialized in JSON-compatible form;
- `task_fingerprint`, a SHA-256 digest of canonical task semantics;
- `structural_fingerprint`, a digest that excludes incidental IDs and continuous
  numeric samples while retaining family, entity topology, goals and action
  shape;
- `difficulty`, with stable categorical dimensions.

The validator will prefer the embedded `task_spec`. Legacy rows without it may
fall back to seed-based task lookup for backward compatibility, but the report
will warn that they are not self-contained. A row with an embedded task must be
validated against that task regardless of current generator output.

## Generator design

Generators will share a small internal interface that produces a candidate
`TaskSpec`, followed by common acceptance checks. Configuration will expose
weighted family selection and difficulty controls without exposing geometric
implementation details.

The first hardening pass expands existing families using supported runtime
semantics:

- hosting QA includes balanced true and false questions and unrelated walls;
- spacing includes static true and false cases, movement in both axes, either
  column as the movable entity, randomized IDs, and distractor columns;
- clearance covers all room walls, both axis signs, randomized movable IDs, and
  distractors;
- opening shift retains all walls but adds distractor openings where scene
  construction permits and validates the final opening position against host
  bounds.

New generated families exercise `inside`, `adjacent`, and multi-goal composition
only where the current mutation/runtime semantics can construct guaranteed
solutions. Unsupported complexity such as arbitrary collision-free planning or
rotation is not synthesized until its runtime invariant can be verified.

Each generated task declares invariants identifying entities/properties that
must remain unchanged. References may contain multiple actions, but unnecessary
actions are rejected by a minimality check for generator-owned references.

## Correctness invariants

For every generated task:

1. schema validation succeeds;
2. all referenced entities exist;
3. every model-visible fact matches the runtime scene;
4. static-positive tasks initially satisfy all goals;
5. static-negative tasks are explicitly labelled as verification tasks and may
   validly assert a false predicate outcome only after the assertion protocol is
   extended; until then they are excluded from SFT export;
6. mutation tasks do not initially satisfy all goals;
7. reference actions execute without errors and satisfy every goal;
8. removing any generator-owned mutation from the reference makes the final
   goal set fail;
9. protected scene state is unchanged;
10. opening host offsets remain within host geometry bounds;
11. all numeric values are finite and within configured ranges.

Because the current `assert` contract represents desired predicates rather than
boolean QA answers, balanced false QA must not be exported using a misleading
successful assertion. The implementation will either add an explicit expected
boolean to the assertion contract with full runtime/eval support or limit the
initial export to mutation and positive-verification examples. The preferred
implementation is the explicit expected-boolean contract.

## Validation and diagnostics

Hard failures will include:

- prompt/task scene mismatch;
- prompt collisions where identical visible inputs map to incompatible task
  semantics;
- non-self-contained records in strict mode;
- invalid references, leaked goals, non-minimal reference mutations or protected
  state changes;
- out-of-bounds opening geometry;
- train/held-out exact fingerprint overlap when split manifests are supplied.

The report will add:

- exact, assistant-payload and structural duplicate rates;
- task and structural fingerprint counts;
- family, predicate, expected-outcome, entity-count, goal-count and action-count
  histograms;
- direction/movable-role coverage for mutation families;
- difficulty distribution;
- prompt collision count;
- self-contained-record rate.

Warnings remain suitable for sampling concerns; semantic contradictions are hard
failures.

## Held-out evaluation

Changing only the random seed remains the IID split. Evaluation configuration
adds named split kinds:

- `iid`: unseen numeric samples;
- `language_ood`: held-out instruction templates;
- `structure_ood`: held-out topology/distractor/direction combinations;
- `boundary`: values around predicate tolerance boundaries;
- `compositional`: familiar predicates in unseen multi-goal combinations.

Train and evaluation manifests record fingerprints. Exact semantic overlap is a
hard failure; structural overlap is reported per split and prohibited for
`structure_ood` and `compositional`.

## Compatibility

- Existing public generator entry points remain available.
- Existing JSONL rows remain loadable in non-strict validation mode.
- The assistant payload stays `bim_gym_actions_v1` unless expected-boolean assert
  requires a versioned `v2`; v1 parsing remains supported.
- Existing deterministic behavior for a fixed configuration is retained.
- No new third-party runtime dependency is required.

## Testing strategy

Development follows red-green-refactor. Tests cover:

- canonical prompt serialization and round-trip task reconstruction;
- prompt/task counterfactual consistency;
- legacy and self-contained JSONL validation;
- fingerprint stability under dictionary ordering and sensitivity to semantic
  changes;
- structural fingerprint invariance to IDs and continuous samples;
- balanced generator coverage over deterministic large samples;
- reference solvability, initial-state leakage, mutation minimality and protected
  state;
- tolerance boundary cases;
- split overlap detection;
- end-to-end export, reload and strict validation for multiple seeds.

Property-style deterministic seed sweeps will cover at least 10,000 generated
tasks in the slow validation suite, while the default unit suite uses a smaller
sample.

## Acceptance criteria

- `python -m pytest -q` passes.
- Strict validation of freshly exported rows reports zero hard failures and a
  self-contained rate of 1.0 for seeds 0 and 1.
- No prompt collisions occur in a 10,000-task sweep.
- Every mutation family covers all supported direction/movable-role variants.
- Balanced boolean-capable QA families remain within 45/55 outcome balance over
  10,000 tasks.
- Exact train/held-out fingerprint overlap is zero for every named OOD split.
- Reference baseline protocol validity and success remain 1.0 on exported
  positive/mutation tasks and on boolean QA after its assertion semantics are
  implemented.

## Delivery sequence

1. canonical model-visible context and self-contained metadata;
2. strict validator, fingerprints and diagnostics;
3. explicit boolean QA semantics;
4. generator symmetry, negatives and distractors;
5. reference minimality and protected-state checks;
6. named held-out splits and overlap manifests;
7. deterministic large-seed validation and documentation.
