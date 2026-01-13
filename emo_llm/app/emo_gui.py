import threading
import tkinter as tk
from tkinter import ttk, messagebox

# 같은 폴더의 emo_core.py 필요
from emo_core import EmotionPipeline

def to_pb(v: float) -> int:
    # [-1,1] -> [0,100]
    return int(max(0, min(100, (v + 1.0) * 50.0)))

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Emotion-Net GUI")
        self.geometry("780x560")
        self.resizable(True, True)

        self.pipe = None
        self._build_widgets()
        self._load_pipeline_async()

    # ---------- UI ----------
    def _build_widgets(self):
        # 입력 영역
        frm_in = ttk.LabelFrame(self, text="입력")
        frm_in.pack(fill="x", padx=10, pady=8)

        self.txt = tk.Text(frm_in, height=5, wrap="word")
        self.txt.pack(fill="x", padx=8, pady=8)

        frm_ctrl = ttk.Frame(frm_in)
        frm_ctrl.pack(fill="x", padx=8, pady=(0,8))
        ttk.Label(frm_ctrl, text="세션 ID:").pack(side="left")
        self.session_var = tk.StringVar(value="default")
        ttk.Entry(frm_ctrl, textvariable=self.session_var, width=20).pack(side="left", padx=6)

        self.btn_analyze = ttk.Button(frm_ctrl, text="분석", command=self.on_analyze, state="disabled")
        self.btn_analyze.pack(side="left", padx=4)
        ttk.Button(frm_ctrl, text="입력 지우기", command=lambda: self.txt.delete("1.0", "end")).pack(side="left", padx=4)

        self.btn_reset = ttk.Button(frm_ctrl, text="세션 상태 초기화", command=self.on_reset_session, state="disabled")
        self.btn_reset.pack(side="left", padx=4)

        # 상태 표시
        self.status = tk.StringVar(value="모델 로딩 중…")
        ttk.Label(self, textvariable=self.status, anchor="w").pack(fill="x", padx=12)

        # 결과 영역
        frm_res = ttk.Frame(self)
        frm_res.pack(fill="both", expand=True, padx=10, pady=6)
        frm_res.columnconfigure(0, weight=1)
        frm_res.columnconfigure(1, weight=1)
        frm_res.rowconfigure(0, weight=1)

        # 감정 벡터 패널
        self.panel_em = ttk.LabelFrame(frm_res, text="감정 벡터 [-1..1]")
        self.panel_em.grid(row=0, column=0, sticky="nsew", padx=(0,8), pady=4)
        self._build_emotion_panel(self.panel_em)

        # 네트워크/디코딩/출력 패널
        self.panel_misc = ttk.LabelFrame(frm_res, text="네트워크/디코딩/출력")
        self.panel_misc.grid(row=0, column=1, sticky="nsew", padx=(8,0), pady=4)
        self._build_misc_panel(self.panel_misc)

    def _build_emotion_panel(self, parent):
        self.vars_now, self.vars_mix = {}, {}
        self.pbars_now, self.pbars_mix = {}, {}

        def row(r, label, key):
            ttk.Label(parent, text=label, width=6).grid(row=r, column=0, sticky="w", padx=8, pady=4)

            vnow = tk.StringVar(value="now: 0.00")
            pm_now = ttk.Progressbar(parent, length=210, mode="determinate", maximum=100)
            pm_now.grid(row=r, column=1, sticky="w", padx=6)
            ttk.Label(parent, textvariable=vnow, width=12).grid(row=r, column=2, sticky="w", padx=4)

            vmix = tk.StringVar(value="mix: 0.00")
            pm_mix = ttk.Progressbar(parent, length=210, mode="determinate", maximum=100)
            pm_mix.grid(row=r, column=3, sticky="w", padx=6)
            ttk.Label(parent, textvariable=vmix, width=12).grid(row=r, column=4, sticky="w", padx=4)

            self.vars_now[key] = vnow; self.pbars_now[key] = pm_now
            self.vars_mix[key] = vmix; self.pbars_mix[key] = pm_mix

        ttk.Label(parent, text="(왼쪽: now / 오른쪽: mixed)").grid(row=0, column=0, columnspan=5, sticky="w", padx=8, pady=(6,2))
        row(1, "DA",  "DA")
        row(2, "5HT", "5HT")
        row(3, "NE",  "NE")
        row(4, "ACh", "ACh")
        for c in range(5):
            parent.columnconfigure(c, weight=0)
        parent.rowconfigure(5, weight=1)

    def _build_misc_panel(self, parent):
        # 네트워크 통계
        lf_net = ttk.LabelFrame(parent, text="네트워크")
        lf_net.pack(fill="x", padx=8, pady=(8,4))
        self.lbl_sE   = tk.StringVar(value="sE: -")
        self.lbl_sI   = tk.StringVar(value="sI: -")
        self.lbl_theta= tk.StringVar(value="θ: -")
        self.lbl_gain = tk.StringVar(value="gain: -")
        ttk.Label(lf_net, textvariable=self.lbl_sE).pack(anchor="w")
        ttk.Label(lf_net, textvariable=self.lbl_sI).pack(anchor="w")
        ttk.Label(lf_net, textvariable=self.lbl_theta).pack(anchor="w")
        ttk.Label(lf_net, textvariable=self.lbl_gain).pack(anchor="w")

        # 디코딩 하이퍼
        lf_dec = ttk.LabelFrame(parent, text="디코딩")
        lf_dec.pack(fill="x", padx=8, pady=(4,8))
        self.lbl_temp = tk.StringVar(value="temperature: -")
        self.lbl_topp = tk.StringVar(value="top_p: -")
        ttk.Label(lf_dec, textvariable=self.lbl_temp).pack(anchor="w")
        ttk.Label(lf_dec, textvariable=self.lbl_topp).pack(anchor="w")

        # 출력 노드
        lf_out = ttk.LabelFrame(parent, text="출력 노드")
        lf_out.pack(fill="x", padx=8, pady=(4,8))
        self.out_var = tk.StringVar(value="r_out: -")
        ttk.Label(lf_out, textvariable=self.out_var, font=("TkDefaultFont", 11, "bold")).pack(anchor="w")

        parent.pack_propagate(False)

    # ---------- 파이프라인 로딩/실행 ----------
    def _load_pipeline_async(self):
        def _load():
            try:
                self.pipe = EmotionPipeline()  # emo_core의 기본 모델 사용
                self.after(0, self._on_loaded)
            except Exception as e:
                self.after(0, lambda: self._on_load_failed(e))
        threading.Thread(target=_load, daemon=True).start()

    def _on_loaded(self):
        self.status.set("준비 완료. 텍스트를 입력하고 [분석]을 눌러줘.")
        self.btn_analyze.configure(state="normal")
        self.btn_reset.configure(state="normal")

    def _on_load_failed(self, e: Exception):
        self.status.set("로딩 실패")
        messagebox.showerror("오류", f"모델 로딩에 실패했습니다.\n{e}")
        self.btn_analyze.configure(state="disabled")
        self.btn_reset.configure(state="disabled")

    def on_analyze(self):
        if not self.pipe:
            return
        text = self.txt.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("알림", "텍스트를 입력해줘.")
            return
        sid = self.session_var.get().strip() or "default"
        self.status.set("분석 중…")
        self.btn_analyze.configure(state="disabled")

        def _run():
            try:
                res = self.pipe.process(text, sid)
                self.after(0, lambda: self._update_result(res))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("오류", str(e)))
            finally:
                self.after(0, lambda: (self.btn_analyze.configure(state="normal"),
                                       self.status.set("완료")))
        threading.Thread(target=_run, daemon=True).start()

    def on_reset_session(self):
        try:
            sid = self.session_var.get().strip() or "default"
            # 세션 잔향 제거
            self.pipe.mem._s.pop(sid, None)
            self.status.set(f"세션 '{sid}' 상태 초기화됨.")
        except Exception as e:
            messagebox.showerror("오류", str(e))

    # ---------- 결과 반영 ----------
    def _update_result(self, res: dict):
        now = res.get("emotion_now", [0,0,0,0])
        mix = res.get("emotion_mix", [0,0,0,0])
        labels = ["DA","5HT","NE","ACh"]

        for i,k in enumerate(labels):
            v_now = float(now[i]); v_mix = float(mix[i])
            self.vars_now[k].set(f"now: {v_now:+.2f}")
            self.pbars_now[k]["value"] = to_pb(v_now)
            self.vars_mix[k].set(f"mix: {v_mix:+.2f}")
            self.pbars_mix[k]["value"] = to_pb(v_mix)

        st = res.get("net_stats", {})
        self.lbl_sE.set(   f"sE: {st.get('sE','-'):.3f}"    if 'sE' in st    else "sE: -")
        self.lbl_sI.set(   f"sI: {st.get('sI','-'):.3f}"    if 'sI' in st    else "sI: -")
        self.lbl_theta.set(f"θ: {st.get('theta','-'):.3f}"  if 'theta' in st else "θ: -")
        self.lbl_gain.set( f"gain: {st.get('gain','-'):.3f}"if 'gain' in st  else "gain: -")

        dec = res.get("decode", {})
        if dec:
            self.lbl_temp.set(f"temperature: {dec.get('temperature',0):.3f}")
            self.lbl_topp.set(f"top_p: {dec.get('top_p',0):.3f}")

        self.out_var.set(f"r_out: {res.get('net_out', 0):+.3f}")

if __name__ == "__main__":
    app = App()
    app.mainloop()
