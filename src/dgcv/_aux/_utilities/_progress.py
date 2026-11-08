"""
package: dgcv - Differential Geometry with Complex Variables

module: dgcv._aux._utilities._progress.py


---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/

Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations


def _timed_progress_call(
    fn,
    *,
    timed: bool,
    threshold_s: float,
    step_desc: str,
    continue_desc: str | None,
    progress_message: str | None,
    _on_timed_update=None,
):
    if not timed:
        return fn()

    fired = {"v": False}
    timer = {"obj": None}
    use_signal = False

    try:
        import threading as _threading

        use_signal = _threading.current_thread() is _threading.main_thread()
    except Exception:
        use_signal = False

    def _emit_update():
        if fired["v"]:
            return
        fired["v"] = True
        if callable(_on_timed_update):
            try:
                _on_timed_update()
            except Exception:
                pass
        print(f"Update: {step_desc}.")
        if progress_message:
            print(progress_message)

    if use_signal:
        prev_handler = None
        prev_itimer = None

        def _handler(signum, frame):
            _emit_update()

        try:
            import signal

            prev_handler = signal.getsignal(signal.SIGALRM)
            prev_itimer = signal.getitimer(signal.ITIMER_REAL)
        except Exception:
            prev_handler = None
            prev_itimer = None

        try:
            import signal

            signal.signal(signal.SIGALRM, _handler)
            signal.setitimer(signal.ITIMER_REAL, max(0.0, float(threshold_s)))
        except Exception:
            use_signal = False
            try:
                import signal

                if prev_handler is not None:
                    signal.signal(signal.SIGALRM, prev_handler)
                if prev_itimer is not None:
                    signal.setitimer(signal.ITIMER_REAL, prev_itimer[0], prev_itimer[1])
            except Exception:
                pass

    if not use_signal:
        try:
            import threading

            t = threading.Timer(max(0.0, float(threshold_s)), _emit_update)
            timer["obj"] = t
            t.daemon = True
            t.start()
        except Exception:
            timer["obj"] = None

    try:
        out = fn()
    finally:
        if use_signal:
            try:
                import signal

                signal.setitimer(signal.ITIMER_REAL, 0.0)
                if prev_handler is not None:
                    signal.signal(signal.SIGALRM, prev_handler)
                if prev_itimer is not None:
                    signal.setitimer(signal.ITIMER_REAL, prev_itimer[0], prev_itimer[1])
            except Exception:
                pass
        else:
            try:
                t = timer["obj"]
                if t is not None:
                    t.cancel()
            except Exception:
                pass

    if fired["v"] and continue_desc:
        print(f"Continuing to {continue_desc}.")
    return out
