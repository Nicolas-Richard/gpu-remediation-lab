# DCGM metrics simulator

This local-only fixture supplies the XID metric missing from the fake GPU exporter's output in kind. One simulator pod runs on each fake-GPU worker. The AWS tests do not use it; they use NVIDIA's `dcgm-exporter` and inject field values with `dcgmi test`.

The simulator listens on port `9400`:

| Request | Result |
| --- | --- |
| `GET /healthz` | Report process health |
| `GET /metrics` | Return Prometheus metrics |
| `GET /state` | Return the simulated observation |
| `PUT /state` with `{"xid": 79}` | Emit an XID value |
| `DELETE /state` | Omit the XID series |

An absent series represents unknown health, `0` represents explicit healthy evidence, and `79` exercises a critical XID. Restarting the pod resets the state to absent.

Run the complete local lifecycle with:

```bash
make demo-up
make test-dcgm-fault-recovery
make demo-down
```

The test configures the GPU node health controller with `--health-source=dcgm-simulator`; simulated observations are never combined with the AWS exporter or annotation source.
