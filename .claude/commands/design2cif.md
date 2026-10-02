---
description: Interview for design intent, inspect artifacts, and build or edit a validated annotated CIF
---

# `/design2cif`

Route `$ARGUMENTS` by the supplied input format:

- Prose, raw structures, or existing annotated CIFs: use
  [rfd4-proteina-annotated-cif](../../skills/rfd4-proteina-annotated-cif/SKILL.md).
- Current `input`/`conditions` YAML or JSON: follow the
  [condition-spec documentation](../../docs/workflow/condition-spec.md).

Follow the selected workflow, clarify material design ambiguities, and
validate the resulting CIF. Do not create an intermediate JSON/YAML for prose or
structure inputs unless requested.
