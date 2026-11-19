"""SupervisorNode: runs batches of episodes without an operator (data generation
with the autopilot, policy evaluation). For each selected model and target it
resets, optionally records, waits for success (held for `hold` s) or timeout,
then moves on; prints a summary and shuts down at the end."""
from __future__ import annotations

import numpy as np

from ..bus import Bus, Node
from ..msgs import Catalog, Event, TaskStatus


class SupervisorNode(Node):
    def __init__(self, bus: Bus, models: list[int], targets: list[str] | None, episodes: int,
                 max_time: float, record: bool, hold: float = 0.5):
        super().__init__(bus, "supervisor")
        self.models, self.targets_filter = list(models), targets
        self.episodes, self.max_time, self.record, self.hold = episodes, max_time, record, hold
        self.catalog: Catalog | None = None
        self.status: TaskStatus | None = None
        self.jobs: list[tuple[int, int, int]] = []
        self.phase = "next_model"
        self.results: list[dict] = []
        self.pub = self.create_publisher("/episode/events")
        self.create_subscription("/task/catalog", lambda c: setattr(self, "catalog", c))
        self.create_subscription("/task/status", lambda s: setattr(self, "status", s))
        self.create_timer(bus.dt, self.update, priority=10)

    def update(self) -> None:
        if self.phase == "next_model":
            if not self.models:
                self._finish()
                return
            m = self.models.pop(0)
            self.pub.publish(Event("set_scenario", {"model": m, "target": 0}))
            names = self.catalog.targets
            idx = [i for i, n in enumerate(names) if not self.targets_filter or n in self.targets_filter]
            self.jobs = [(m, i, e) for i in idx for e in range(self.episodes)]
            self.phase = "start"
        if self.phase == "start":
            if not self.jobs:
                self.phase = "next_model"
                return
            self.job = self.jobs.pop(0)
            self.pub.publish(Event("set_scenario", {"model": self.job[0], "target": self.job[1]}))
            if self.record:
                self.pub.publish(Event("record_start"))
            self.t_start, self.success_since = self.now, None
            self.phase = "run"
            return
        if self.phase == "run":
            st = self.status
            if st is None or st.stamp is None or st.stamp <= self.t_start:
                return
            if st.success:
                self.success_since = self.success_since if self.success_since is not None else self.now
                if self.now - self.success_since >= self.hold:
                    self._end(True)
                    return
            else:
                self.success_since = None
            if self.now - self.t_start >= self.max_time:
                self._end(False)

    def _end(self, ok: bool) -> None:
        if self.record:
            self.pub.publish(Event("record_stop"))
        cat, st = self.catalog, self.status
        t = (self.success_since if ok else self.now) - self.t_start
        r = {"model": cat.models[cat.model_index], "target": st.target_name, "episode": self.job[2],
             "success": ok, "time": round(t, 2), "final_dist": round(st.target_dist, 2)}
        self.results.append(r)
        print(f"{r['model']:12s} {r['target']:12s} ep{r['episode']}  {'OK  ' if ok else 'FAIL'}"
              f"  t={r['time']:6.1f}s  dist={r['final_dist']:.1f} mm", flush=True)
        self.phase = "start"

    def _finish(self) -> None:
        if self.results:
            ok = [r for r in self.results if r["success"]]
            print(f"\n{len(ok)}/{len(self.results)} successful"
                  + (f", mean time {np.mean([r['time'] for r in ok]):.1f}s" if ok else ""))
        self.pub.publish(Event("shutdown"))
        self.bus.shutdown()
