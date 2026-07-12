import os
import json
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch


@dataclass
class PcSuccessRateControl:
    # =========================
    # =========================
    direct_low: float = 0.20
    direct_high: float = 0.80

    acc_margin: float = 0.02

    min_fmt_n: int = 16

    gate_B_value: float = 2.0

    # =========================
    # =========================
    force_B_by_batch_imbalance: bool = True
    pi_hi: float = 0.98
    pi_lo: float = 0.02
    ratio_hi: float = 50.0
    forced_B_value: float = 5.0

    # =========================
    # =========================
    log_root: Optional[str] = None
    log_filename: str = "pc_metrics_simple.jsonl"
    flush_every: int = 1
    add_timestamp: bool = True
    _log_path: Optional[str] = None
    _log_count: int = 0

    # =========================
    # =========================
    last_B: float = 0.0

    def __post_init__(self):
        if self.log_root is not None:
            os.makedirs(self.log_root, exist_ok=True)
            self._log_path = os.path.join(self.log_root, self.log_filename)
        self.last_B = 0.0

    def _append_jsonl(self, record: Dict[str, Any]):
        if not self._log_path:
            return
        rec = dict(record)
        if self.add_timestamp:
            rec["_ts"] = time.time()
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._log_count += 1
            if self.flush_every <= 1 or (self._log_count % self.flush_every == 0):
                f.flush()

    def update_from_metrics(
        self,
        accuracy,
        is_thinking,
        format_ok,
        device=None,
        *,
        log_step: Optional[int] = None,
        write_log: bool = True,
    ) -> Dict[str, Any]:
        """
         B（ reward ）：

        （ batch  format_ok ）：
          -  nothink(=direct)   nothink  =>  nothink => B > 0
          -  nothink(=direct)   nothink  =>  think   => B < 0
          -  B = 0（deadband，）

          - think （pi_batch>pi_hi / <pi_lo  ratio_hi） =>  B 
          -  batch  format_ok  => B  last_B
        """
        if device is None:
            device = torch.device("cpu")

        is_correct = torch.as_tensor(accuracy, device=device) > 0.5
        is_high = torch.as_tensor(is_thinking, device=device) > 0.5  # True=think/high
        fmt = torch.as_tensor(format_ok, device=device) > 0.5

        bn_high = int(torch.sum(fmt & is_high).item())
        bn_direct = int(torch.sum(fmt & (~is_high)).item())
        bok_high = int(torch.sum(fmt & is_correct & is_high).item())
        bok_direct = int(torch.sum(fmt & is_correct & (~is_high)).item())

        denom_bd = bn_high + bn_direct

        batch_format_ok = float(torch.sum(fmt).item())
        batch_format_err = float(torch.sum(~fmt).item())
        format_ok_rate = float(batch_format_ok / (fmt.numel() + 1e-8))

        if denom_bd > 0:
            pi_batch_think = float(bn_high / (denom_bd + 1e-8))
            direct_share = float(bn_direct / (denom_bd + 1e-8))
        else:
            pi_batch_think = float("nan")
            direct_share = float("nan")

        acc_high = float(bok_high / (bn_high + 1e-8)) if bn_high > 0 else float("nan")
        acc_direct = float(bok_direct / (bn_direct + 1e-8)) if bn_direct > 0 else float("nan")

        B = 0.0
        B_raw = 0.0
        B_overridden = False
        override_reason = ""

        if denom_bd == 0:
            B = float(self.last_B)
            B_overridden = True
            override_reason = "no_format_ok_in_batch_use_last_B"
        else:
            if denom_bd >= int(self.min_fmt_n) and (bn_high > 0) and (bn_direct > 0):
                if (acc_direct - acc_high) > float(self.acc_margin) and direct_share < float(self.direct_low):
                    B = +abs(float(self.gate_B_value))
                    override_reason = (
                        f"simple_gate:direct_better_and_rare "
                        f"(acc_d={acc_direct:.3f},acc_h={acc_high:.3f},direct_share={direct_share:.3f})"
                    )
                elif (acc_high - acc_direct) > float(self.acc_margin) and direct_share > float(self.direct_high):
                    B = -abs(float(self.gate_B_value))
                    override_reason = (
                        f"simple_gate:direct_worse_and_common "
                        f"(acc_d={acc_direct:.3f},acc_h={acc_high:.3f},direct_share={direct_share:.3f})"
                    )
                else:
                    B = 0.0
                    override_reason = "simple_gate:no_action"
            else:
                B = 0.0
                override_reason = "simple_gate:insufficient_batch_or_one_side_empty"

            B_raw = float(B)

            if self.force_B_by_batch_imbalance:
                if pi_batch_think == pi_batch_think:  # not nan
                    if pi_batch_think > float(self.pi_hi):
                        B = +abs(float(self.forced_B_value))
                        B_overridden = True
                        override_reason = f"extreme_guard:pi_batch>{self.pi_hi}"
                    elif pi_batch_think < float(self.pi_lo):
                        B = -abs(float(self.forced_B_value))
                        B_overridden = True
                        override_reason = f"extreme_guard:pi_batch<{self.pi_lo}"

                if (not B_overridden) and self.ratio_hi is not None:
                    n_h = float(bn_high)
                    n_d = float(bn_direct)
                    if n_h > float(self.ratio_hi) * (n_d + 1.0):
                        B = +abs(float(self.forced_B_value))
                        B_overridden = True
                        override_reason = f"extreme_guard:n_high>ratio_hi*(n_direct+1), ratio_hi={self.ratio_hi}"
                    elif n_d > float(self.ratio_hi) * (n_h + 1.0):
                        B = -abs(float(self.forced_B_value))
                        B_overridden = True
                        override_reason = f"extreme_guard:n_direct>ratio_hi*(n_high+1), ratio_hi={self.ratio_hi}"

        B = float(B)
        bias_label = "encourage_nothink" if B > 0 else ("encourage_think" if B < 0 else "balanced")

        out: Dict[str, Any] = {
            "actor/bias_B": float(B),
            "actor/bias_label": bias_label,

            "actor/bias_B_raw": float(B_raw),
            "actor/bias_B_overridden": float(B_overridden),
            "actor/bias_B_override_reason": override_reason,

            "actor/batch_pi_think": float(pi_batch_think) if pi_batch_think == pi_batch_think else -1.0,
            "actor/batch_direct_share": float(direct_share) if direct_share == direct_share else -1.0,
            "actor/batch_acc_think": float(acc_high) if acc_high == acc_high else -1.0,
            "actor/batch_acc_direct": float(acc_direct) if acc_direct == acc_direct else -1.0,

            "actor/batch_n_high": float(bn_high),
            "actor/batch_n_direct": float(bn_direct),
            "actor/batch_ok_high": float(bok_high),
            "actor/batch_ok_direct": float(bok_direct),

            "actor/batch_format_ok": batch_format_ok,
            "actor/batch_format_err": batch_format_err,
            "actor/format_ok_rate": format_ok_rate,

            "actor/direct_low": float(self.direct_low),
            "actor/direct_high": float(self.direct_high),
            "actor/acc_margin": float(self.acc_margin),
            "actor/min_fmt_n": float(self.min_fmt_n),
            "actor/gate_B_value": float(self.gate_B_value),

            "actor/force_B_by_batch_imbalance": float(self.force_B_by_batch_imbalance),
            "actor/pi_hi": float(self.pi_hi),
            "actor/pi_lo": float(self.pi_lo),
            "actor/ratio_hi": float(self.ratio_hi),
            "actor/forced_B_value": float(self.forced_B_value),
        }

        if log_step is not None:
            out["actor/step"] = int(log_step)

        self.last_B = float(B)

        if write_log:
            self._append_jsonl(out)

        return out
