# trainpulse (Python client)

Zero-dependency client for the [TrainPulse](../..) training diagnostics daemon.

```sh
pip install trainpulse   # or: pip install -e clients/python
```

## Report metrics from a training loop — two lines

```python
from trainpulse import TrainPulseClient

tp = TrainPulseClient()                      # daemon on 127.0.0.1:9876
for step, batch in enumerate(loader):
    with tp.step(global_step=step, tokens=batch.numel()):
        loss = train_step(batch)
```

`step()` times the block and reports `step_time_ms` / `tokens_per_sec`.
Reporting **never raises** — if the daemon is down, training is unaffected.

Richer metrics go through `report()`:

```python
tp.report(step_time_ms=184.2, tokens_per_sec=72500, mfu=0.42,
          data_wait_ms=12.4, all_reduce_wait_ms=16.0, world_size=8)
```

## Read health and optimization recommendations

```python
tp.health()               # 0-100 health score
tp.snapshot()             # full telemetry + signals + diagnoses + utilization
tp.recommendations()      # cluster utilization + tuning recommendations
```

## Auto-apply safe optimizations with Tuner

The daemon marks convergence-neutral knobs (e.g. dataloader workers) as
`auto_applicable`. `Tuner.suggested()` returns those; everything that could
change training semantics (batch geometry, precision) stays advisory:

```python
from trainpulse import TrainPulseClient, Tuner

tuner = Tuner(TrainPulseClient())
num_workers = tuner.suggested("dataloader_workers", current=num_workers)

for rec in tuner.advisories():   # needs a human decision
    print(rec["id"], rec["impact"], rec.get("suggested"))
```

## Auth

If the daemon runs with `-auth-token`:

```python
tp = TrainPulseClient("http://gpu-host:9876", token="...")
```
