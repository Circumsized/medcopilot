## Summary
- 

## Validation Checklist
- [ ] `MOCK_LLM=1 PYTHONPATH=.. pytest tests/ -v -m "not slow"` passes
- [ ] `ruff check .` passes
- [ ] `python -m compileall -q .` passes
- [ ] `MOCK_LLM=1 PYTHONPATH=.. python scripts/eval_golden.py` passes
- [ ] `MOCK_LLM=1 PYTHONPATH=.. python scripts/emergency_eval.py` passes
- [ ] `MOCK_LLM=1 PYTHONPATH=.. python scripts/policy_check.py` passes
- [ ] `MOCK_LLM=1 PYTHONPATH=.. python scripts/recovery_drill.py` passes
- [ ] No API keys, credentials, or PHI/PII included in code, fixtures, or logs
