Task: {task_id}
The failing cluster is grouped by first missing milestone: `{first_missing_milestone}`
({n_failures} failed episode(s) share this).

{failure_summary}

{success_summary}

Explain the ONE most likely causal failure mechanism that explains why the
representative failure never reached `{first_missing_milestone}`, while the
successful reference did. Do not propose any fix, tool, or code -- that is
a separate stage.

Respond as JSON only, matching this shape:
{{
  "mechanism": "<one or two sentences>",
  "first_missing_milestone": "{first_missing_milestone}",
  "divergence_step": <int, the step where things first went wrong>,
  "evidence": "<what in the summaries above supports this>"
}}
