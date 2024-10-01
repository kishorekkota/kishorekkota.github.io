## JCONF Dalla Sessions Attended

### Memory Management

Different type of Garbage Collection and how to tune them.

Stop the world events and how to avoid them.

- [ ] [Java Memory Management](https://www.youtube.com/watch?v=7cGyJ5wvUe0)
- Choose Right Grabage Collector.
- Tune Heapsize to balance between frequent minor GC and unneeded promotions to old Generation.
- Avoid Full GC by tuning the heap size.
  * Older Generation is Full.
- Types of GC in Java 23.
  * G1 GC - Default Since Java 9.
  * ZGC - Low Latency GC. Designed for Multi-TB Heaps.
  * Shenandoah - Low Latency GC. Designed for Multi-TB Heaps.
  * Parallel GC - Designed for better throughput. But can take larger pauses impacting latency.
  * Serial GC - Designed for small heaps.

  