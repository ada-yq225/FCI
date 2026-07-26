[Timing: 1:25]

The historical FCI schedule begins with a complete undirected graph, performs a PC-style adjacency search, stores separating sets, and orients unshielded colliders. The central non-local correction is Possible-D-SEP: for each endpoint order, FCI searches vertices reachable along paths whose intermediate triples are colliders or members of triangles. This computable set is a superset of the graphical D-SEP information that may contain a separating set missed by local neighborhoods. When a new independence is found, FCI removes the edge, resets marks, and runs its orientation phase again. Theorem 6.4 in the 2000 book is an oracle-scope soundness result under Faithfulness and correct conditional-independence decisions. It returns a partially oriented inducing-path graph; the same source explicitly did not establish that its historical rule schedule was maximally informative. Modern PAG orientation under Zhang’s later rule set is therefore a separate ingredient, not something retroactively attributed to the book.

[Sources]
- Spirtes, Glymour, and Scheines, *Causation, Prediction, and Search*, 2nd ed. (2000), Possible-D-SEP and Fast Causal Inference Algorithm, Chapter 6 §6.7, printed pp. 144–145.
- Spirtes et al. (2000), Theorem 6.4 and the immediately following paragraph, printed p. 145.
- Zhang (2008), *Artificial Intelligence* 172(16–17), §§3–4, pp. 1873–1896, DOI `10.1016/j.artint.2008.08.001`.
- `reports/research/claim_evidence_matrix.csv`, claim IDs `FCI-PDSEP`, `FCI-STAGES`, `FCI-SOUND`, and `FCI-INCOMPLETE`.
[/Sources]