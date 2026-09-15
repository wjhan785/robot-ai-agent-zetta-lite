Diagnosis for this cluster (round {round_id}):
mechanism: {mechanism}
first_missing_milestone: {first_missing_milestone}
divergence_step: {divergence_step}
evidence: {evidence}

Feedback from the previous attempt (if any): {previous_feedback}

Available tools (you may ONLY use these, with these argument names):
{tool_catalog}

Propose exactly one critic+recovery bundle as JSON only, matching this
shape (see schemas/bundle.schema.json for the full grammar):
{{
  "id": "<short unique id>",
  "critic": ["<feature>", "<op>", <value>],
  "recovery": [
    {{"tool": "<tool name>", "args": {{}}}}
  ],
  "reentry": ["<feature>", "<op>", <value>]
}}

`critic` and `reentry` may also be {{"all": [...]}} or {{"any": [...]}} of
nested rules. The critic should fire as early as possible at or after
divergence_step (a stall-detecting critic usually needs a few steps to
confirm a stall before it can safely fire -- that's expected). The
recovery should be the shortest sequence of tool calls that plausibly
reaches `first_missing_milestone`. `reentry` should describe when it is
safe to hand control back to the frozen policy.
