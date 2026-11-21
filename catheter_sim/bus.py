"""A minimal ROS2-style publish/subscribe layer (single process, no ROS).

Concepts mirror rclpy:
  Node                 a component with publishers, subscriptions and timers
  Topic                a named channel; `latched=True` behaves like a ROS2
                       transient-local topic (late subscribers get the last message)
  Timer                periodic callback on the *sim clock*
  Bus (executor)       owns the clock and runs every node's timers each tick

Delivery is synchronous and in-process: `publish()` calls each subscriber's
callback immediately. Keep subscription callbacks cheap (store the message);
do heavy work in timers. Timers fire in `priority` order inside a tick, which
makes a run fully deterministic.

Time: every message carries `stamp` (sim time the data was valid) and `seq`
(a bus-wide counter, strictly increasing in the order data was produced), so
messages with equal stamps are still totally ordered. See msgs.py.

Clock modes:
  realtime  the bus sleeps so sim time tracks wall-clock time (teleop, live demos)
  lockstep  no sleeping: run as fast as possible, every node sees every tick
            (policy evaluation, data generation)
"""
from __future__ import annotations

import time
from typing import Any, Callable


class Topic:
    def __init__(self, name: str):
        self.name = name
        self.latched = False
        self.last: Any = None
        self.count = 0
        self.subscribers: list[Callable[[Any], None]] = []


class Publisher:
    def __init__(self, bus: "Bus", topic: Topic):
        self.bus, self.topic = bus, topic

    def publish(self, msg: Any) -> None:
        if getattr(msg, "stamp", 0.0) is None:
            msg.stamp = self.bus.now
        if getattr(msg, "seq", 0) is None:
            msg.seq = self.bus.next_seq()
        self.topic.last = msg
        self.topic.count += 1
        for cb in list(self.topic.subscribers):
            cb(msg)


class Timer:
    def __init__(self, node: "Node", period_ticks: int, callback: Callable[[], None],
                 priority: int, order: int):
        self.node, self.period, self.callback = node, max(period_ticks, 1), callback
        self.priority, self.order = priority, order


class Node:
    def __init__(self, bus: "Bus", name: str):
        self.bus, self.name = bus, name
        bus.nodes.append(self)

    def create_publisher(self, topic: str, latched: bool = False) -> Publisher:
        t = self.bus.topic(topic)
        t.latched = t.latched or latched
        return Publisher(self.bus, t)

    def create_subscription(self, topic: str, callback: Callable[[Any], None]) -> None:
        t = self.bus.topic(topic)
        t.subscribers.append(callback)
        if t.latched and t.last is not None:
            callback(t.last)

    def create_timer(self, period_s: float, callback: Callable[[], None], priority: int = 50) -> Timer:
        tm = Timer(self, round(period_s / self.bus.dt), callback, priority, len(self.bus.timers))
        self.bus.timers.append(tm)
        self.bus.timers.sort(key=lambda x: (x.priority, x.order))
        return tm

    @property
    def now(self) -> float:
        return self.bus.now

    def on_shutdown(self) -> None:
        """Override to flush files, close windows, ..."""


class Bus:
    def __init__(self, rate_hz: int = 60, realtime: bool = True):
        self.rate_hz = rate_hz
        self.dt = 1.0 / rate_hz
        self.realtime = realtime
        self.tick = 0
        self.seq = 0                    # next sequence number
        self.topics: dict[str, Topic] = {}
        self.nodes: list[Node] = []
        self.timers: list[Timer] = []
        self._stop = False
        self.wall_rate = 0.0            # achieved ticks per wall-clock second

    @property
    def now(self) -> float:
        return self.tick * self.dt

    def next_seq(self) -> int:
        self.seq += 1
        return self.seq - 1

    def topic(self, name: str) -> Topic:
        if name not in self.topics:
            self.topics[name] = Topic(name)
        return self.topics[name]

    def shutdown(self) -> None:
        self._stop = True

    def spin_once(self) -> None:
        for tm in self.timers:
            if self.tick % tm.period == 0:
                tm.callback()
                if self._stop:
                    break
        self.tick += 1

    def spin(self, max_time: float | None = None) -> None:
        t0 = time.perf_counter()
        start_tick = self.tick
        rate_t, rate_n = t0, 0
        try:
            while not self._stop:
                if max_time is not None and self.now >= max_time:
                    break
                self.spin_once()
                rate_n += 1
                wall = time.perf_counter()
                if wall - rate_t >= 0.5:
                    self.wall_rate, rate_t, rate_n = rate_n / (wall - rate_t), wall, 0
                if self.realtime:
                    due = t0 + (self.tick - start_tick) * self.dt
                    if due > wall:
                        time.sleep(due - wall)
                    elif wall - due > 0.25:       # fell behind (slow step): don't try to catch up
                        t0, start_tick = wall, self.tick
        finally:
            for n in self.nodes:
                n.on_shutdown()
