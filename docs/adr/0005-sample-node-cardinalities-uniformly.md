---
status: accepted
---

# Sample each node cardinality uniformly from two through five

Every node independently draws its state count uniformly from
\(\{2,3,4,5\}\). Conditional on the node count, every resulting cardinality
vector therefore has equal probability and no node label receives a special
cardinality distribution.

With at most three parents, a node has at most \(5^3=125\) conditional rows
and \(625\) stored probability entries. This admits genuinely multivalued
worlds while keeping exact CPT construction and inference bounded at the
per-node level.
