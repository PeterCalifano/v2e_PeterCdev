# Findings: Event Stream Ordering and I/O

Scope: the contract between event generation and everything that consumes it —
packet ordering, global stream monotonicity, writer timestamp ranges, and the
checks that are supposed to protect them.

Conventions and severity definitions: [`README.md`](README.md).

---

## STREAM-001

**Latency breaks global timestamp monotonicity**

- **Severity:** High
- **Status:** Confirmed (probe)
- **Where:** `v2ecore/emulator.py`, `_apply_latency_jitter_and_sort()`

### Issue

Latency offsets are added to event timestamps and the result is sorted — but
the sort covers only the current frame packet:

```python
sorted_idx = torch.argsort(events[:, 0])
```

Each packet is then returned to the caller and written immediately. An event
delayed out of packet `k` can therefore land after events already emitted from
packet `k+1`.

Both latency paths are affected: `iebcs_latency_jitter_model` and
`iebcs_contrast_latency_model`.

### Why it is an issue

Monotonic timestamps across the complete output stream is a stated correctness
invariant of this repository — `development_core.md` lists it, and every
consumer relies on it. Breaking it has consequences beyond untidy ordering:

- **Writers.** HDF5, AEDAT-2, AEDAT-4 and text output all receive events in
  emission order. AEDAT readers in particular treat backward time steps as
  corruption or as a timestamp wrap.
- **Frame/event index attribution.** The HDF5 `frame_ev_idx` mapping assumes
  event index order matches time order. Once it does not, slicing events by
  frame returns a set that is neither time-contiguous nor complete.
- **Downstream consumers.** `EventDataGenerationLib` canonicalises these streams
  for `event-based-centroiding`; a non-monotonic input propagates into datasets
  built for model training.

The defect is a direct consequence of the design: a per-packet sort cannot
order events against packets that have not been generated yet. It needs a
release policy, not a bigger sort.

### Evidence

Five alternating frames at 2 ms spacing, latency mean 800 µs, jitter 400 µs,
fixed seed, packets concatenated in emission order:

```text
total events: 468
backward steps across concatenated packets: 3
largest backward step: -1128.9 us
```

### Suggested fix

Hold delayed events in an emulator-owned pending queue and release only what is
safe under a watermark:

1. On each frame, push newly delayed events into a queue keyed by their delayed
   timestamp.
2. Emit only events whose timestamp is `<= t_previous` (the newest time for
   which no further event can still arrive), which for a bounded latency is
   `t_frame - max_latency`.
3. Provide an explicit finalisation call that flushes the remaining tail through
   the same conversion and writer path, so no event is lost at end of stream.

Keep signal/noise labels in the same queue entry as the event so the label
permutation cannot drift from the events it describes.

The alternative — clamping latency to the packet interval — preserves ordering
without a queue, but changes the model: it makes the maximum representable
latency depend on the frame rate. If that route is taken, say so in
`README_error_models_extensions.md` and reject `--iebcs_latency_mean_us` values
that exceed the frame interval, rather than silently truncating the
distribution.

Note that fixing this requires deciding what `generate_events()` returns for a
frame whose events are still pending, which is an API change for direct callers.
That decision belongs with the fix, not after it.

---

## STREAM-002

**Writer timestamp ranges wrap without an explicit contract**

- **Severity:** Medium
- **Status:** Confirmed (inspection)
- **Where:** `v2ecore/emulator.py` HDF5 event conversion;
  `v2ecore/output/aedat2_output.py`

### Issue

Timestamps are narrowed at several points with no range check:

- Event packets carry float32 seconds.
- Frame timestamps are cast to float32 before microsecond integer conversion.
- HDF5 stores `uint32` microseconds — wrapping after ~4294.97 s (71.6 min).
- AEDAT-2 paths using signed `int32` microseconds wrap after ~2147.48 s
  (35.8 min).

### Why it is an issue

Each limit is a silent cliff. A recording that crosses it produces a file whose
timestamps restart from zero mid-stream, which downstream tooling will read as
either a corrupt file or — worse — a valid one describing different times than
the run actually produced.

The float32 issue is subtler and bites earlier. float32 has ~24 bits of
mantissa, so at `t = 1000 s` the spacing is ~6e-5 s: sub-100-µs event timing is
no longer representable, and distinct events collapse onto identical
timestamps. A model whose whole purpose is microsecond-accurate timing loses
that accuracy after a few minutes of stream time, with no diagnostic.

There is no documented maximum recording duration anywhere in the repository, so
a user has no way to know any of these boundaries exist.

### Suggested fix

Make the contract explicit and enforced:

1. State a supported maximum recording duration per writer in
   `README_error_models_extensions.md` and in each writer's docstring.
2. Keep frame and event timestamps in float64 internally, narrowing only at the
   serialisation boundary where the target dtype is known.
3. Raise a clear error at the writer when a timestamp exceeds the format's
   range, naming the format and its limit, instead of allowing the wrap.
4. Add tests immediately below and above each boundary
   (`2147.48 s`, `4294.97 s`).

If long recordings are a real requirement, the follow-on design question is
whether to store a per-file time origin offset — which is cheap and removes the
absolute-time limit — but that is a format change and should be decided
separately.

---

## STREAM-003

**Monotonicity check warns per packet and cannot see the real defect**

- **Severity:** Low
- **Status:** Confirmed (inspection)
- **Where:** `v2ecore/emulator.py`, end of `generate_events()`

### Issue

After converting events to NumPy, the emulator checks its own output:

```python
if np.any(np.diff(timestamps) < 0):
    idx = np.argwhere(np.diff(timestamps) < 0)
    v2e_logger.warning(f'nonmonotonic timestamp(s) at indices {idx}')
```

This inspects one packet in isolation. Since `_apply_latency_jitter_and_sort()`
has just sorted that same packet, the check is nearly always vacuous — while the
actual defect, [STREAM-001](#stream-001), is invisible to it by construction.

### Why it is an issue

A check that cannot fail in the case it appears to guard is worse than no check:
it reads like coverage. Anyone auditing the code sees an explicit monotonicity
guard next to the event output and reasonably concludes ordering is validated.

The warning is also a poor failure mode for a violated invariant — it logs and
continues, so a corrupted stream is still written to disk and still returned to
the caller.

### Suggested fix

Make the check span what it claims to protect:

- Track the last emitted timestamp on the emulator and compare each packet's
  first event against it, so cross-packet regressions are detected.
- Escalate from `warning` to a raised error under a strict mode, so tests and
  batch pipelines fail loudly rather than producing a bad file.
- Once [STREAM-001](#stream-001) is fixed, keep the check as a permanent
  invariant assertion rather than deleting it — it becomes meaningful at that
  point.

Until the cross-packet fix lands, add a comment stating the check's scope, so
its current limits are not mistaken for a guarantee.
