# NIST AI Risk Management Framework: Core Functions (Summary Notes)

These notes summarise, in the author's own words, the four core functions of
the NIST AI Risk Management Framework, with emphasis on how they apply to an
explainability and fairness audit tool for a predictive model.

## Govern

This function concerns the policies, processes, and culture an organisation
puts in place around AI risk. In practice, this means having a defined
process for reviewing model explainability and fairness reports before a
model is deployed or before significant changes are made, and assigning
clear ownership for acting on flagged issues.

## Map

This function concerns understanding the context in which an AI system
operates: who is affected by its decisions, what the consequences of an
incorrect decision are, and which groups might be disproportionately
affected. For a credit-scoring model, mapping would identify applicants as
the primary affected group, note that an incorrect rejection can have
significant financial consequences, and identify any characteristics
(such as the applicant_group attribute in this example) that should be
monitored for disparities.

## Measure

This function concerns the actual measurement of risk, including model
performance, robustness, and fairness metrics. Feature-attribution outputs
(such as SHAP values) and group-level fairness metrics (such as demographic
parity difference and equalized odds difference) are direct examples of
"Measure" activities. The framework emphasises that measurement should be
ongoing, not a one-off exercise at launch.

## Manage

This function concerns acting on the results of measurement: prioritising
identified risks, deciding on mitigations (which might include retraining,
adjusting decision thresholds per group, or adding human review for
borderline cases), and documenting the rationale for those decisions. A
flagged fairness metric should lead to a documented management decision,
even if that decision is "monitor further" rather than an immediate model
change.

## Relevance to this tool

An explainability and fairness audit tool maps most directly onto the
"Measure" function, producing the metrics that feed into "Manage" decisions.
For the tool's outputs to be useful within the framework, they should be
presented in a way that supports a documented decision, for example by
clearly distinguishing between metrics that are within an organisation's
defined tolerance and those that require escalation.
