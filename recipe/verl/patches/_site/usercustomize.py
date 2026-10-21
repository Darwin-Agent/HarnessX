# Auto-loaded by every Python process via PYTHONPATH.
# Patches SGLang stop string support for verl's hybrid mode.
#
# Guard: only applies patches if sglang is ALREADY imported (avoids triggering
# heavy sglang import in Ray workers that don't need it). The scheduler
# subprocess imports sglang before this point, so patches apply correctly.
import os
import sys

if os.environ.get("HX_ROOT") and "sglang.srt.managers.scheduler" in sys.modules:
    try:
        from sglang.srt.sampling.sampling_params import SamplingParams
        from sglang.srt.managers.scheduler import Scheduler

        _orig_normalize = SamplingParams.normalize

        def _patched_normalize(self, tokenizer):
            _orig_normalize(self, tokenizer)
            if self.stop_strs:
                max_len = self.stop_str_max_len
                for s in self.stop_strs:
                    max_len = max(max_len, len(s) * 2)
                self.stop_str_max_len = max_len

        SamplingParams.normalize = _patched_normalize

        _orig_init_tokenizer = Scheduler.init_tokenizer

        def _patched_init_tokenizer(self):
            _orig_init_tokenizer(self)
            if self.tokenizer is None:
                mp = os.environ.get("MODEL_DIR", "")
                if mp:
                    try:
                        from transformers import AutoTokenizer
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            mp, trust_remote_code=True
                        )
                        if not hasattr(self.tokenizer, "additional_stop_token_ids"):
                            self.tokenizer.additional_stop_token_ids = None
                    except Exception:
                        pass

        Scheduler.init_tokenizer = _patched_init_tokenizer
    except (ImportError, Exception):
        pass
