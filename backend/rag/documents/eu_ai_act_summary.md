# EU AI Act: Risk Categories and Relevance to Credit Scoring (Summary Notes)

These notes summarise, in the author's own words, the parts of the EU AI Act
most relevant to an automated credit-scoring system. They are intended as
retrieval context for an internal explainability tool, not as legal advice.

## Risk-based classification

The Act sorts AI systems into tiers based on the risk they pose to
fundamental rights and safety. Systems used to evaluate the creditworthiness
of natural persons, or to establish their credit score, fall into the
high-risk category. High-risk classification triggers a set of obligations
for the provider and the deployer of the system.

## Obligations for high-risk systems

Providers of high-risk systems are expected to maintain a risk-management
process across the system's lifecycle, ensure the quality of training,
validation, and testing datasets so that they are relevant, representative,
and to the best extent possible free of errors, and keep technical
documentation that allows authorities to assess compliance.

Deployers are expected to use the system in line with the instructions for
use, monitor its operation, and keep logs where appropriate. Where a
deployer identifies that the system presents a risk, they have an obligation
to inform the provider and relevant authorities.

## Human oversight

High-risk systems must be designed so that natural persons can oversee their
operation, including the ability to understand the system's output, decide
not to use it in a particular situation, or override or reverse an output.
This principle is often referred to as human oversight or human-in-the-loop
control, and is a recurring theme across most AI governance frameworks, not
only the EU AI Act.

## Bias and non-discrimination

The Act places emphasis on examining training data for possible biases that
could lead to discrimination prohibited under EU law, including
discrimination based on characteristics protected under equality law. Where
a system's outputs differ materially between groups defined by a protected
characteristic, this is treated as a signal that the dataset, model, or both
may require further examination, even if no single feature in the model
explicitly encodes the protected characteristic.

## Transparency towards affected persons

Where an AI system is used to make or materially inform a decision about a
person, for example whether to approve a loan, that person should be made
aware that an AI system is involved, and should be given a meaningful
explanation of the role the system played in the decision, to the extent
feasible.

## Relevance to feature-attribution explanations (such as SHAP)

Feature-attribution methods support several of the obligations above. They
help demonstrate that a model's decisions are driven by factors that are
plausibly related to creditworthiness, such as income and credit history,
rather than by a protected characteristic. They also support the
transparency obligation by providing a basis for explaining individual
decisions to affected persons in accessible language.

A large difference in approval rates or error rates between groups defined
by a protected characteristic, even where that characteristic is not itself
a model feature, should be treated as a prompt for further investigation
under the bias and non-discrimination principles described above, rather
than as conclusive evidence of unlawful discrimination on its own.
