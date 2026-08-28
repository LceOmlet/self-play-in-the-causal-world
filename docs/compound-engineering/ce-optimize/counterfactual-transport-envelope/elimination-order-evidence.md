# Exact elimination-order evidence

The frozen distribution profiler compares the current reverse-topological
paired elimination with an exact min-fill ordering over the same symbolic
factors. It emits aggregate ratios only.

- Total intermediate cells: 2/7 unresolved instances fall to at most 10% of
  the current order, one more falls to at most 25%, one to at most 50%, and one
  to at most 75%. Two show no material reduction.
- Peak intermediate table: 2/7 fall to at most 10%, one more to at most 25%,
  two to at most 75%, and two show no material reduction.
- The candidate changes only associative elimination order. It neither drops
  a factor nor changes a local response-coupling owner.

This is the first remaining hypothesis with measured end-to-end formulation
size reduction on the unresolved class. It advances to semantic parity and
full frozen-cohort measurement.

