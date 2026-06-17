# Human-in-the-Loop Oversight and Explanation Quality (Summary Notes)

These notes summarise general principles, in the author's own words, for
designing the human oversight and explanation components of an AI
decision-support tool.

## Why human oversight matters

Automated decision systems are most useful when they support a human
decision-maker rather than fully replace their judgement, particularly for
decisions with significant consequences for the affected person, such as
credit, employment, or insurance decisions. Human oversight allows a person
with relevant context to catch cases where a model's output seems
inconsistent with other information available to them, and provides a
mechanism for affected individuals to contest a decision.

## What makes an explanation useful to a human reviewer

A useful explanation for a human reviewer should connect the model's output
to the specific input values for that case, in terms the reviewer
understands without needing to interpret raw numerical output. For example,
stating that an application was associated with a lower approval probability
primarily because of a high debt-to-income ratio is more useful to a
reviewer than stating only that "feature 3 contributed -0.12".

An explanation should also avoid implying more certainty than the underlying
metric supports. A feature-attribution value indicates the direction and
relative size of a factor's influence on a specific prediction; it is not a
causal claim and is specific to the model being explained, not a general
statement about the real-world relationship between the feature and the
outcome.

## Feedback loops

When a human reviewer disagrees with or flags an explanation as unhelpful or
potentially misleading, recording that feedback creates a useful signal.
Over time, patterns in reviewer feedback, such as repeated disagreement on
cases involving a particular feature or group, can indicate that the
underlying model or the explanation method itself warrants closer
examination. This is consistent with the "Measure" and "Manage" functions
described in risk-management frameworks: reviewer feedback is itself a
measurement that should inform management decisions.

## Communicating uncertainty and limitations

Any automated audit tool should be transparent about its own limitations.
For example, a fairness metric computed on a held-out sample may not be
representative of the full population, illustrative thresholds used to flag
metrics are not the same as legal or regulatory thresholds, and an
explanation grounded in retrieved reference text reflects that text's
framing, not an authoritative legal determination. Surfacing these caveats
alongside the explanation supports appropriate reliance by the human
reviewer.
