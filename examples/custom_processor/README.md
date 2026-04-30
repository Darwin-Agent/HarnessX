# Custom Processor Example

This example shows how to add a custom processor with `_target_`.

## Files

- `harness_config.yaml`: example harness config using a custom processor.
- `processors/simple_guard.py`: `SimpleGuardProcessor` implementation.

## Run

From repo root:

```bash
hx -d examples/custom_processor/harness_config.yaml
```

## Key idea

Any importable class that subclasses `harnessx.core.processor.MultiHookProcessor`
can be referenced in `processors` via `_target_`.

For `step_start` processors, read user input from `event.messages or event.raw_messages`.
In minimal harnesses, `messages` may be empty before context-assembly processors run.
