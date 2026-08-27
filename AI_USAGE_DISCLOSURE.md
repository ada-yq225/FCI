# AI Usage Disclosure

## Scope and central declaration

This statement covers the `fci-engine` software repository, the Tennessee STAR
case-study pipeline, and the submitted final report. Generative AI was used
only as a programming aid during software development. It was **not used in
the preparation of the final written report**.

The AI system was a technical assistant, not an author, researcher, source of
evidence, or decision maker. I retained control over the project's design,
implementation, validation, interpretation, and written submission.

## My independent contribution

I independently conceived and carried out the substantive academic work of the
project. My contribution includes the following.

### Research design and methodological decisions

- Defined the research problem: implementing and evaluating auditable FCI and
  FCI+ causal-discovery workflows that return Partial Ancestral Graphs (PAGs)
  under possible latent confounding or selection.
- Chose the research scope, the theoretical sources to study, the comparison
  criteria, and the boundary between source-faithful algorithm validation and
  finite-sample applied analysis.
- Determined the configuration profiles, conditional-independence testing
  strategy, accuracy and stability metrics, sensitivity analyses, and the
  interpretation limits applied to the learned graphs.
- Selected the Tennessee STAR application, its public source, the cohort and
  variables used, the three analysis panels, and the temporal and substantive
  checks used to interpret the findings.

### Software and data-analysis work

- Designed the package structure and implemented, integrated, reviewed, and
  maintained the `fci_engine` codebase, including PAG representations,
  conditional-independence test interfaces, skeleton discovery, Possible-D-SEP,
  FCI+ D-SEP search, orientation rules, result objects, diagnostics, and
  export functions.
- Designed the reproducible STAR workflow: data acquisition, validation,
  cohort definition, variable coding, panel construction, analysis execution,
  bootstrap and variable-order audits, sensitivity runs, and report-artifact
  generation.
- Chose and executed the verification strategy, including unit and integration
  tests, exact-oracle graph checks, seeded finite-sample simulations,
  comparisons with `causal-learn` and R `pcalg`, scaling experiments, and
  reproducibility checks.
- Ran the analyses, inspected outputs and failures, interpreted disagreements
  across methods and settings, and decided which results and limitations were
  appropriate to report.

### Final written report

- Independently read and checked the literature cited in the report.
- Independently wrote the abstract, main text, captions, tables,
  methodological explanations, results interpretation, discussion,
  limitations, conclusion, and bibliography.
- Independently selected the evidence presented, prepared the LaTeX source,
  and checked the final PDF and all statements within it.

I am solely responsible for all research decisions, code accepted into the
repository, numerical results, citations, analysis, and conclusions.

## Generative AI used for software development only

I used **OpenAI Codex (GPT-5.6 Sol)**, developed by OpenAI, and **Claude
Fable 5**, developed by Anthropic, as programming assistants during software
development. Their role was limited to technical assistance: discussing
implementation approaches, explaining Python or library behaviour, suggesting
or debugging code, and identifying possible tests or validation checks. I did
not treat suggestions as correct by default; I reviewed them, modified or
rejected them where appropriate, and verified accepted changes by running the
project's checks.

For the Tennessee STAR case study, Codex assistance included parts of the data
cleaning and reproducibility software. The relevant coding work included:

- implementing checks when downloading the documented public Harvard Dataverse
  export, including the expected table dimensions;
- implementing transformations that restrict the dataset to pupils with a
  kindergarten class assignment and create the specified analysis cohorts;
- recoding documented class assignment, gender, ethnicity, free-lunch status,
  school context, and teacher-experience variables;
- constructing entry-age, kindergarten-achievement, Grade-3-achievement, and
  Grade-3-observation variables using the documented score and binning rules;
- handling missing values separately for each pre-specified analysis panel,
  factorising the discrete inputs, and writing reproducible processed tables;
- maintaining diagnostics, result exports, tests, type checks, and code
  refactoring for the FCI/FCI+ implementation and case-study workflow.

The AI system did not choose the dataset, variables, coding rules, statistical
settings, research protocol, or interpretation. It did not execute the
analysis, generate empirical results, fabricate data, or provide unverified
claims used as evidence. The numerical results in the project are produced by
the committed scripts and are subject to the project's test and validation
procedures.

The STAR data used in this work come from the documented public Harvard
Dataverse release. No confidential data, credentials, or private participant
information were supplied to the AI assistant.

## Final report: no generative-AI use

No generative AI was used to prepare the final written report. This exclusion
is complete: I did not use AI to draft, rewrite, paraphrase, summarise,
translate, condense, proofread, or edit any part of the final report. Nor did
I use it to create or select citations, figures, tables, results, arguments,
or conclusions for that report.

The report is therefore distinct from the AI-assisted coding process. Its
prose, scholarly judgement, presentation of evidence, and interpretation were
written and checked by me alone. AI output was not used as a scholarly source
or as a substitute for my own analysis.

## Accountability and verification

The repository preserves the code, data-acquisition and data-processing
scripts, automated tests, analysis configurations, result artifacts, and final
report source so that the technical work can be independently inspected. I
accept full responsibility for the project and this disclosure.

This is a factual record of AI use, not a claim that an AI system shares
authorship or responsibility. The applicable course policy takes precedence
over this statement.
