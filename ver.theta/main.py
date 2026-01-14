# -*- coding: utf-8 -*-
"""
[Neuro-Chatbot: ver.theta]
- Fixed K range (membrane potential is clamped).
- ΔW represents connection count and is modulated by NTL.
- Includes GUI, logging, and visualization.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Deque, List, Tuple
import json
import logging
import os
import random
import time
import urllib.request
import urllib.error
import uuid
from collections import deque

import tkinter as tk
from tkinter import ttk

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


Vec4 = Tuple[float, float, float, float]

K_MIN = 0.0
K_MAX = 2.0
CONNECTION_DELTA_SCALE = 3
HISTORY_LIMIT = 200
LLM_DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")


matplotlib.use("TkAgg")


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def ensure_logs_dir(base_dir: str) -> str:
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def parse_ntl_payload(payload: str) -> Vec4:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        start = payload.find("{")
        end = payload.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("LLM JSON payload not found.")
        data = json.loads(payload[start : end + 1])

    values = [
        clamp(float(data.get("D", 0.5)), 0.0, 1.0),
        clamp(float(data.get("S", 0.5)), 0.0, 1.0),
        clamp(float(data.get("NE", 0.5)), 0.0, 1.0),
        clamp(float(data.get("M", 0.5)), 0.0, 1.0),
    ]
    return (values[0], values[1], values[2], values[3])


def fetch_ntl_from_llm(text: str) -> Vec4:
    if not LLM_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    url = f"{LLM_API_BASE.rstrip('/')}/chat/completions"
    system_prompt = (
        "You are an emotion vector extractor. "
        "Return ONLY JSON with keys D, S, NE, M in [0,1]."
    )
    user_prompt = f'문장: "{text}"'
    payload = {
        "model": LLM_DEFAULT_MODEL,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": f"EmoNet-ver-theta/{uuid.uuid4().hex}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"LLM HTTP error: {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("LLM request failed.") from exc

    data = json.loads(raw)
    content = data["choices"][0]["message"]["content"]
    return parse_ntl_payload(content)


@dataclass
class DataBox:
    K: float
    V: Vec4
    trace: List[str]


@dataclass
class Edge:
    src: "Neuron"
    dst: "Neuron"
    W: float

    def send(self, box: DataBox) -> None:
        self.dst._next_inbox.append((box, self))


class Neuron:
    def __init__(self, name: str, kind: str):
        self.name = name
        self.kind = kind
        self.threshold = 0.4
        self.W = 1.0
        self._inbox: Deque[Tuple[DataBox, Edge]] = deque()
        self._next_inbox: Deque[Tuple[DataBox, Edge]] = deque()
        self.outgoing: List[Edge] = []
        self.incoming: List[Edge] = []
        self.refractory = 1
        self.off_ticks = 0

    def connect_to(self, other: "Neuron") -> Edge:
        for edge in self.outgoing:
            if edge.dst is other:
                return edge
        edge = Edge(src=self, dst=other, W=1.0)
        self.outgoing.append(edge)
        other.incoming.append(edge)
        return edge

    def swap_buffer(self) -> None:
        if self._next_inbox:
            self._inbox.extend(self._next_inbox)
            self._next_inbox.clear()

    def tick(self, net: "Network") -> None:
        if self.off_ticks > 0:
            self.off_ticks -= 1
            self._inbox.clear()
            return
        if not self._inbox:
            return

        V_in, K_in = self._combine_inputs()
        trace = self._inbox[-1][0].trace[-5:] + [self.name]

        if K_in < self.threshold:
            self._inbox.clear()
            return

        K_out, V_out, d_conn = self._specific_op(K_in, V_in)
        K_out = clamp(K_out * self.W, K_MIN, K_MAX)

        outbox = DataBox(K=K_out, V=V_out, trace=trace)
        for edge in self.outgoing:
            edge.send(outbox)

        if d_conn != 0:
            self._apply_plasticity(net, d_conn)

        self._inbox.clear()
        self.off_ticks = self.refractory

    def _combine_inputs(self) -> Tuple[Vec4, float]:
        if not self._inbox:
            return (0.5, 0.5, 0.5, 0.5), 0.0

        v_sum = [0.0, 0.0, 0.0, 0.0]
        k_sum = 0.0
        weight_sum = 0.0
        for box, edge in self._inbox:
            for idx in range(4):
                v_sum[idx] += edge.W * box.V[idx]
            k_sum += edge.W * box.K
            weight_sum += abs(edge.W)

        if weight_sum == 0.0:
            return (0.5, 0.5, 0.5, 0.5), 0.0

        V_in = tuple(v / weight_sum for v in v_sum)
        K_in = clamp(k_sum / weight_sum, K_MIN, K_MAX)
        return V_in, K_in

    def _specific_op(self, K_in: float, V_in: Vec4) -> Tuple[float, Vec4, int]:
        D, S, NE, M = V_in
        d_conn = self._delta_connections(D, S, NE, M)

        if self.kind == "exc":
            factor = 1.0 + 0.8 * K_in
            V_out = tuple(clamp(0.5 + (v - 0.5) * factor, 0.0, 1.0) for v in V_in)
            return K_in, V_out, d_conn
        if self.kind == "inh":
            shrink = clamp(0.4 + S * 0.4 - D * 0.1, 0.1, 0.9)
            V_out = tuple(clamp(0.5 + (v - 0.5) * (1.0 - shrink), 0.0, 1.0) for v in V_in)
            return K_in, V_out, d_conn
        return K_in, V_in, d_conn

    def _delta_connections(self, D: float, S: float, NE: float, M: float) -> int:
        gate = (D + NE) - (S + M)
        delta = int(round(gate * CONNECTION_DELTA_SCALE))
        return max(-CONNECTION_DELTA_SCALE, min(CONNECTION_DELTA_SCALE, delta))

    def _apply_plasticity(self, net: "Network", delta: int) -> None:
        if delta > 0:
            candidates = [n for n in net.neurons if n is not self and all(e.dst is not n for e in self.outgoing)]
            random.shuffle(candidates)
            for target in candidates[:delta]:
                self.connect_to(target)
        elif delta < 0:
            if not self.outgoing:
                return
            random.shuffle(self.outgoing)
            remove_count = min(abs(delta), len(self.outgoing))
            for _ in range(remove_count):
                if len(self.outgoing) <= 1:
                    break
                edge = self.outgoing.pop()
                if edge in edge.dst.incoming:
                    edge.dst.incoming.remove(edge)


class Network:
    def __init__(self, neurons: List[Neuron]):
        self.neurons = neurons
        self.source = neurons[0]

    def wire_random(self, p: float = 0.5, seed: int = 42) -> None:
        rng = random.Random(seed)
        for a in self.neurons:
            for b in self.neurons:
                if a is b:
                    continue
                if rng.random() < p:
                    a.connect_to(b)

    def tick(self) -> None:
        for neuron in self.neurons:
            neuron.swap_buffer()
        for neuron in self.neurons:
            neuron.tick(self)

    def inject(self, box: DataBox) -> None:
        dummy = Neuron("Input", "exc")
        edge = Edge(src=dummy, dst=self.source, W=1.0)
        self.source._next_inbox.append((box, edge))

    def connection_counts(self) -> List[int]:
        return [len(n.outgoing) for n in self.neurons]


class ThetaApp:
    def __init__(self, root: tk.Tk, logger: logging.Logger):
        self.root = root
        self.logger = logger
        self.root.title("EmoNet ver.theta")

        self.net = self._build_network()
        self.tick_count = 0

        self.history_ntl: List[Vec4] = []
        self.history_k: List[float] = []
        self.history_conn: List[float] = []

        self._build_ui()
        self._update_plots()

    def _build_network(self) -> Network:
        kinds = ["exc"] * 10 + ["inh"] * 5 + ["reg"] * 5
        neurons = [Neuron(f"N{i}", kind) for i, kind in enumerate(kinds)]
        net = Network(neurons)
        net.wire_random(p=0.4, seed=42)
        return net

    def _build_ui(self) -> None:
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        input_frame = ttk.LabelFrame(main_frame, text="입력")
        input_frame.pack(fill=tk.X)

        ttk.Label(input_frame, text="텍스트").grid(row=0, column=0, sticky=tk.W, padx=4, pady=4)
        self.text_entry = ttk.Entry(input_frame, width=50)
        self.text_entry.grid(row=0, column=1, columnspan=4, sticky=tk.W, padx=4, pady=4)

        labels = ["D", "S", "NE", "M"]
        self.ntl_entries: List[ttk.Entry] = []
        for idx, label in enumerate(labels):
            ttk.Label(input_frame, text=label).grid(row=1, column=idx, padx=4, pady=4)
            entry = ttk.Entry(input_frame, width=8)
            entry.insert(0, "0.5")
            entry.grid(row=2, column=idx, padx=4, pady=4)
            self.ntl_entries.append(entry)

        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=3, column=0, columnspan=5, sticky=tk.W)
        ttk.Button(button_frame, text="Inject", command=self.on_inject).pack(side=tk.LEFT, padx=4)
        ttk.Button(button_frame, text="Tick", command=self.on_tick).pack(side=tk.LEFT, padx=4)
        ttk.Button(button_frame, text="Run 10", command=lambda: self.run_ticks(10)).pack(side=tk.LEFT, padx=4)
        ttk.Button(button_frame, text="LLM 분석", command=self.on_llm_analyze).pack(side=tk.LEFT, padx=4)

        status_frame = ttk.LabelFrame(main_frame, text="상태")
        status_frame.pack(fill=tk.X, pady=6)
        self.status_label = ttk.Label(status_frame, text="Tick: 0")
        self.status_label.pack(anchor=tk.W, padx=4, pady=4)

        fig = Figure(figsize=(7, 4), dpi=100)
        self.ax_ntl = fig.add_subplot(311)
        self.ax_k = fig.add_subplot(312)
        self.ax_conn = fig.add_subplot(313)
        fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(fig, master=main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def on_inject(self) -> None:
        text = self.text_entry.get().strip()
        vec = self._parse_ntl()
        if vec is None:
            return

        box = DataBox(K=1.0, V=vec, trace=["IPT"])
        self.net.inject(box)
        self.logger.info("Inject text='%s' NTL=%s", text, vec)
        self._record_state(vec)
        self._update_status()

    def on_llm_analyze(self) -> None:
        text = self.text_entry.get().strip()
        if not text:
            self.status_label.config(text="텍스트를 입력하세요.")
            return

        try:
            vec = fetch_ntl_from_llm(text)
        except RuntimeError as exc:
            self.status_label.config(text=str(exc))
            return
        except ValueError as exc:
            self.status_label.config(text=str(exc))
            return

        for entry, value in zip(self.ntl_entries, vec):
            entry.delete(0, tk.END)
            entry.insert(0, f"{value:.3f}")

        self.logger.info("LLM NTL text='%s' NTL=%s", text, vec)
        self._record_state(vec)
        self._update_status()

    def on_tick(self) -> None:
        self.net.tick()
        self.tick_count += 1
        avg_k = self._average_k()
        latest_ntl = self.history_ntl[-1] if self.history_ntl else (0.5, 0.5, 0.5, 0.5)
        self._record_state(latest_ntl, avg_k_override=avg_k)
        self.logger.info("Tick=%s avg_K=%.3f connections=%s", self.tick_count, avg_k, self.history_conn[-1])
        self._update_status()

    def run_ticks(self, count: int) -> None:
        for _ in range(count):
            self.on_tick()

    def _parse_ntl(self) -> Vec4 | None:
        values = []
        for entry in self.ntl_entries:
            raw = entry.get().strip()
            try:
                values.append(clamp(float(raw), 0.0, 1.0))
            except ValueError:
                self.status_label.config(text="NTL 입력이 올바르지 않습니다.")
                return None
        return (values[0], values[1], values[2], values[3])

    def _average_k(self) -> float:
        k_values = []
        for neuron in self.net.neurons:
            if neuron._inbox:
                k_values.append(sum(box.K for box, _ in neuron._inbox) / len(neuron._inbox))
        if not k_values:
            return 0.0
        return clamp(sum(k_values) / len(k_values), K_MIN, K_MAX)

    def _record_state(self, vec: Vec4, avg_k_override: float | None = None) -> None:
        if avg_k_override is None:
            avg_k_override = self._average_k()

        conn_counts = self.net.connection_counts()
        avg_conn = sum(conn_counts) / len(conn_counts)

        self.history_ntl.append(vec)
        self.history_k.append(avg_k_override)
        self.history_conn.append(avg_conn)

        if len(self.history_ntl) > HISTORY_LIMIT:
            self.history_ntl = self.history_ntl[-HISTORY_LIMIT:]
            self.history_k = self.history_k[-HISTORY_LIMIT:]
            self.history_conn = self.history_conn[-HISTORY_LIMIT:]

        self._update_plots()

    def _update_status(self) -> None:
        avg_conn = self.history_conn[-1] if self.history_conn else 0.0
        self.status_label.config(text=f"Tick: {self.tick_count} | Avg K: {self.history_k[-1]:.2f} | Avg Conn: {avg_conn:.1f}")

    def _update_plots(self) -> None:
        if not self.history_ntl:
            self.canvas.draw()
            return

        xs = list(range(len(self.history_ntl)))
        ntl_series = list(zip(*self.history_ntl))

        self.ax_ntl.clear()
        for idx, label in enumerate(["D", "S", "NE", "M"]):
            self.ax_ntl.plot(xs, ntl_series[idx], label=label)
        self.ax_ntl.set_ylim(0.0, 1.0)
        self.ax_ntl.set_title("NTL")
        self.ax_ntl.legend(loc="upper right", fontsize=8)

        self.ax_k.clear()
        self.ax_k.plot(xs, self.history_k, color="purple")
        self.ax_k.set_ylim(K_MIN, K_MAX)
        self.ax_k.set_title("Avg K (clamped)")

        self.ax_conn.clear()
        self.ax_conn.plot(xs, self.history_conn, color="green")
        self.ax_conn.set_title("Avg Connection Count (ΔW 의미)")

        self.canvas.draw()


def setup_logger() -> logging.Logger:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = ensure_logs_dir(base_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"run_{timestamp}.log")

    logger = logging.getLogger("theta")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def main() -> None:
    logger = setup_logger()
    root = tk.Tk()
    app = ThetaApp(root, logger)
    logger.info("ver.theta started")
    root.mainloop()


if __name__ == "__main__":
    main()
