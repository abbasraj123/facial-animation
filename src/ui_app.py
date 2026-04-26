import os
import pathlib as plb
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from project_paths import paths, get_data_root, ensure_dirs


class FacialAnimationApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Speech-Driven Facial Animation")
        self.geometry("980x680")
        self.minsize(900, 620)

        self.repo_root = plb.Path(__file__).resolve().parents[1]
        self.data_root = get_data_root(None)
        self.pipeline_paths = paths(self.data_root)
        ensure_dirs(self.pipeline_paths["root"])

        self.selected_video = tk.StringVar()
        self.video_details = tk.StringVar(value="No video selected")
        self.status = tk.StringVar(value="Ready")
        self.stage = tk.StringVar(value="Select an MP4 video and start processing.")
        self.expression_scale = tk.DoubleVar(value=float(os.environ.get("SFA_EXPRESSION_SCALE", "7")))
        self.mouth_scale = tk.DoubleVar(value=float(os.environ.get("SFA_MOUTH_SCALE", "6")))
        self.audio_mouth_strength = tk.DoubleVar(value=float(os.environ.get("SFA_AUDIO_MOUTH_STRENGTH", "1.2")))
        self.comparison_output = tk.StringVar(value="Not generated yet")
        self.output_3d = tk.StringVar(value="Not generated yet")

        self._configure_style()
        self._build_ui()

    def _configure_style(self):
        self.configure(bg="#f3f6fb")
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass

        self.style.configure("App.TFrame", background="#f3f6fb")
        self.style.configure("Card.TFrame", background="#ffffff", relief=tk.FLAT)
        self.style.configure("Header.TLabel", background="#13233a", foreground="#ffffff", font=("Segoe UI", 22, "bold"))
        self.style.configure("HeaderSub.TLabel", background="#13233a", foreground="#cbd5e1", font=("Segoe UI", 10))
        self.style.configure("Title.TLabel", background="#ffffff", foreground="#1e293b", font=("Segoe UI", 12, "bold"))
        self.style.configure("Body.TLabel", background="#ffffff", foreground="#334155", font=("Segoe UI", 9))
        self.style.configure("Muted.TLabel", background="#ffffff", foreground="#64748b", font=("Segoe UI", 9))
        self.style.configure("Status.TLabel", background="#f3f6fb", foreground="#1e293b", font=("Segoe UI", 10, "bold"))
        self.style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=(14, 8))
        self.style.configure("TButton", font=("Segoe UI", 9), padding=(10, 6))
        self.style.configure("TEntry", padding=5)
        self.style.configure("Accent.Horizontal.TProgressbar", troughcolor="#dbeafe", background="#2563eb", bordercolor="#dbeafe")

    def _build_ui(self):
        root = ttk.Frame(self, style="App.TFrame")
        root.pack(fill=tk.BOTH, expand=True)

        header = tk.Frame(root, bg="#13233a", height=104)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        ttk.Label(header, text="Speech-Driven Facial Animation Studio", style="Header.TLabel").pack(anchor=tk.W, padx=26, pady=(18, 2))
        ttk.Label(
            header,
            text="Select a video, tune lip motion, and export comparison plus 3D rendered animation",
            style="HeaderSub.TLabel",
        ).pack(anchor=tk.W, padx=26)

        badge_row = tk.Frame(header, bg="#13233a")
        badge_row.pack(anchor=tk.W, padx=26, pady=(8, 0))
        self._badge(badge_row, "Pretrained CNTK")
        self._badge(badge_row, "Audio-driven lips")
        self._badge(badge_row, "MP4 export")

        body = ttk.Frame(root, style="App.TFrame", padding=16)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(3, weight=1)

        self._build_input_card(body)
        self._build_tuning_card(body)
        self._build_status_card(body)
        self._build_output_card(body)
        self._build_log_card(body)

    def _build_input_card(self, parent):
        card = self._card(parent)
        card.grid(row=0, column=0, sticky=tk.EW, padx=(0, 10), pady=(0, 12))
        card.columnconfigure(0, weight=1)

        ttk.Label(card, text="1. Select Input Video", style="Title.TLabel").grid(row=0, column=0, sticky=tk.W, pady=(0, 8))
        ttk.Label(card, text="Choose an MP4 file. The app copies external files into data/ before processing.", style="Muted.TLabel").grid(
            row=1, column=0, sticky=tk.W, pady=(0, 10)
        )

        row = ttk.Frame(card, style="Card.TFrame")
        row.grid(row=2, column=0, sticky=tk.EW)
        row.columnconfigure(0, weight=1)

        self.file_entry = ttk.Entry(row, textvariable=self.selected_video)
        self.file_entry.grid(row=0, column=0, sticky=tk.EW)
        ttk.Button(row, text="Browse", command=self.browse_video).grid(row=0, column=1, padx=(8, 0))

        info = tk.Frame(card, bg="#f8fafc", padx=10, pady=8)
        info.grid(row=3, column=0, sticky=tk.EW, pady=(10, 0))
        tk.Label(info, textvariable=self.video_details, bg="#f8fafc", fg="#475569", font=("Segoe UI", 9)).pack(anchor=tk.W)

    def _build_tuning_card(self, parent):
        card = self._card(parent)
        card.grid(row=0, column=1, sticky=tk.NSEW, pady=(0, 12))
        for col in range(3):
            card.columnconfigure(col, weight=1)

        ttk.Label(card, text="Render Tuning", style="Title.TLabel").grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 8))
        ttk.Label(card, text="Higher values make lip and expression movement stronger.", style="Muted.TLabel").grid(
            row=1, column=0, columnspan=3, sticky=tk.W, pady=(0, 10)
        )

        self._add_slider(card, "Expression", self.expression_scale, 2, 12, 2)
        self._add_slider(card, "Mouth", self.mouth_scale, 2, 12, 3)
        self._add_slider(card, "Audio Mouth", self.audio_mouth_strength, 0, 2, 4)

        presets = ttk.Frame(card, style="Card.TFrame")
        presets.grid(row=5, column=0, columnspan=3, sticky=tk.W, pady=(12, 0))
        ttk.Button(presets, text="Natural", command=lambda: self.apply_preset(5, 4, 0.7)).pack(side=tk.LEFT)
        ttk.Button(presets, text="Balanced", command=lambda: self.apply_preset(7, 6, 1.2)).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(presets, text="Strong Lips", command=lambda: self.apply_preset(9, 8, 1.6)).pack(side=tk.LEFT, padx=(6, 0))

    def _build_status_card(self, parent):
        card = ttk.Frame(parent, style="App.TFrame")
        card.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=(0, 12))
        card.columnconfigure(0, weight=1)

        action_row = ttk.Frame(card, style="App.TFrame")
        action_row.grid(row=0, column=0, sticky=tk.EW)
        action_row.columnconfigure(0, weight=1)

        ttk.Label(action_row, textvariable=self.status, style="Status.TLabel").grid(row=0, column=0, sticky=tk.W)
        self.run_btn = ttk.Button(action_row, text="Run Inference + Render", style="Primary.TButton", command=self.start_run)
        self.run_btn.grid(row=0, column=1, padx=(12, 0))

        self.progress = ttk.Progressbar(card, mode="determinate", maximum=100, style="Accent.Horizontal.TProgressbar")
        self.progress.grid(row=1, column=0, sticky=tk.EW, pady=(8, 2))

        stage_label = ttk.Label(card, textvariable=self.stage, background="#f3f6fb", foreground="#475569")
        stage_label.grid(row=2, column=0, sticky=tk.W)

    def _build_output_card(self, parent):
        card = self._card(parent)
        card.grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=(0, 12))
        card.columnconfigure(1, weight=1)

        ttk.Label(card, text="2. Generated Output", style="Title.TLabel").grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))

        ttk.Label(card, text="Comparison video", style="Body.TLabel").grid(row=1, column=0, sticky=tk.W, pady=3)
        ttk.Label(card, textvariable=self.comparison_output, style="Muted.TLabel").grid(row=1, column=1, sticky=tk.EW, padx=10)
        self.open_compare_file_btn = ttk.Button(card, text="Open File", command=self.open_comparison_file)
        self.open_compare_file_btn.grid(row=1, column=2, padx=(8, 0))

        ttk.Label(card, text="3D-only video", style="Body.TLabel").grid(row=2, column=0, sticky=tk.W, pady=3)
        ttk.Label(card, textvariable=self.output_3d, style="Muted.TLabel").grid(row=2, column=1, sticky=tk.EW, padx=10)
        self.open_3d_file_btn = ttk.Button(card, text="Open File", command=self.open_3d_file)
        self.open_3d_file_btn.grid(row=2, column=2, padx=(8, 0))

        folder_row = ttk.Frame(card, style="Card.TFrame")
        folder_row.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(10, 0))
        ttk.Button(folder_row, text="Open Comparison Folder", command=self.open_comparison_folder).pack(side=tk.LEFT)
        ttk.Button(folder_row, text="Open 3D Folder", command=self.open_3d_folder).pack(side=tk.LEFT, padx=(8, 0))

        self._set_output_buttons(False)

    def _build_log_card(self, parent):
        card = self._card(parent)
        card.grid(row=3, column=0, columnspan=2, sticky=tk.NSEW)
        card.rowconfigure(1, weight=1)
        card.columnconfigure(0, weight=1)

        ttk.Label(card, text="Processing Log", style="Title.TLabel").grid(row=0, column=0, sticky=tk.W, pady=(0, 8))

        log_wrap = ttk.Frame(card, style="Card.TFrame")
        log_wrap.grid(row=1, column=0, sticky=tk.NSEW)
        log_wrap.rowconfigure(0, weight=1)
        log_wrap.columnconfigure(0, weight=1)

        self.log = tk.Text(log_wrap, height=12, wrap=tk.WORD, bg="#0f172a", fg="#e2e8f0", insertbackground="#ffffff")
        self.log.grid(row=0, column=0, sticky=tk.NSEW)

        scroll = ttk.Scrollbar(log_wrap, orient=tk.VERTICAL, command=self.log.yview)
        scroll.grid(row=0, column=1, sticky=tk.NS)
        self.log.configure(yscrollcommand=scroll.set)

    def _card(self, parent):
        frame = ttk.Frame(parent, style="Card.TFrame", padding=14)
        return frame

    def _badge(self, parent, text):
        label = tk.Label(parent, text=text, bg="#213957", fg="#e2e8f0", font=("Segoe UI", 8), padx=10, pady=3)
        label.pack(side=tk.LEFT, padx=(0, 8))

    def _add_slider(self, parent, label, variable, from_, to, row):
        frame = ttk.Frame(parent, style="Card.TFrame")
        frame.grid(row=row, column=0, columnspan=3, sticky=tk.EW, pady=(4, 0))
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text=label, style="Body.TLabel", width=13).grid(row=0, column=0, sticky=tk.W)
        scale = tk.Scale(
            frame,
            variable=variable,
            from_=from_,
            to=to,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            showvalue=False,
            bg="#ffffff",
            highlightthickness=0,
            troughcolor="#dbeafe",
            activebackground="#2563eb",
        )
        scale.grid(row=0, column=1, sticky=tk.EW, padx=8)
        value = ttk.Label(frame, textvariable=variable, style="Muted.TLabel", width=5)
        value.grid(row=0, column=2, sticky=tk.E)

    def apply_preset(self, expression, mouth, audio_mouth):
        self.expression_scale.set(expression)
        self.mouth_scale.set(mouth)
        self.audio_mouth_strength.set(audio_mouth)
        self.stage.set("Preset applied. Run again to render with these values.")

    def browse_video(self):
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("MP4 video", "*.mp4"), ("All files", "*.*")],
        )
        if path:
            self.selected_video.set(path)
            video = plb.Path(path)
            size_mb = video.stat().st_size / (1024.0 * 1024.0)
            self.video_details.set("{} | {:.2f} MB | {}".format(video.name, size_mb, video.parent))
            self.stage.set("Selected: " + video.name)

    def append_log(self, text):
        self.log.insert(tk.END, text)
        self.log.see(tk.END)

    def set_running(self, running):
        self.run_btn.configure(state=tk.DISABLED if running else tk.NORMAL)
        if running:
            self._set_output_buttons(False)

    def _set_output_buttons(self, enabled):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.open_compare_file_btn.configure(state=state)
        self.open_3d_file_btn.configure(state=state)

    def start_run(self):
        video = plb.Path(self.selected_video.get().strip('"'))
        if not video.exists() or video.suffix.lower() != ".mp4":
            messagebox.showerror("Invalid video", "Please select a valid .mp4 video file.")
            return

        self.set_running(True)
        self.status.set("Processing")
        self.stage.set("Preparing input video...")
        self.progress["value"] = 0
        self.comparison_output.set("Not generated yet")
        self.output_3d.set("Not generated yet")
        self.log.delete("1.0", tk.END)

        thread = threading.Thread(target=self.run_pipeline, args=(video,), daemon=True)
        thread.start()

    def run_pipeline(self, source_video):
        try:
            data_video = self.pipeline_paths["root"] / source_video.name
            if source_video.resolve() != data_video.resolve():
                ensure_dirs(data_video.parent)
                self.after(0, self.append_log, "Copying video to data folder...\n")
                shutil.copy2(str(source_video), str(data_video))

            env = os.environ.copy()
            env["SFA_EXPRESSION_SCALE"] = "{:.2f}".format(self.expression_scale.get())
            env["SFA_MOUTH_SCALE"] = "{:.2f}".format(self.mouth_scale.get())
            env["SFA_AUDIO_MOUTH_STRENGTH"] = "{:.2f}".format(self.audio_mouth_strength.get())

            commands = [
                (
                    "Running pretrained inference...",
                    10,
                    55,
                    [sys.executable, str(self.repo_root / "src" / "infer_pretrained.py"), "--data-root", str(self.data_root), "--only-video", source_video.name],
                ),
                (
                    "Rendering comparison and 3D videos...",
                    60,
                    95,
                    [sys.executable, str(self.repo_root / "src" / "shape_renderer.py"), "--data-root", str(self.data_root), "--only-video", source_video.name],
                ),
            ]

            for stage_text, start_progress, end_progress, command in commands:
                self.after(0, self.stage.set, stage_text)
                self.after(0, self._set_progress, start_progress)
                self.after(0, self.append_log, "\nRunning: " + " ".join(command) + "\n")
                self._run_command(command, env)
                self.after(0, self._set_progress, end_progress)

            video_name = source_video.stem + ".mp4"
            output_compare = self.pipeline_paths["outputs"] / "rendered_videos" / "default" / video_name
            output_3d = self.pipeline_paths["outputs"] / "rendered_videos_3d" / "default" / video_name

            self.after(0, self.append_log, "\nDone.\n")
            self.after(0, self.append_log, "Comparison video: {}\n".format(output_compare))
            self.after(0, self.append_log, "3D-only video: {}\n".format(output_3d))
            self.after(0, self.comparison_output.set, str(output_compare))
            self.after(0, self.output_3d.set, str(output_3d))
            self.after(0, self.status.set, "Completed")
            self.after(0, self.stage.set, "Output videos generated successfully.")
            self.after(0, self._set_progress, 100)
            self.after(0, self._set_output_buttons, True)
        except Exception as exc:
            self.after(0, self.append_log, "\nError: {}\n".format(exc))
            self.after(0, self.status.set, "Failed")
            self.after(0, self.stage.set, "Processing failed. See log for details.")
            self.after(0, messagebox.showerror, "Run failed", str(exc))
        finally:
            self.after(0, self.set_running, False)

    def _run_command(self, command, env):
        process = subprocess.Popen(
            command,
            cwd=str(self.repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        for line in process.stdout:
            self.after(0, self.append_log, line)
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError("Command failed with exit code {}".format(return_code))

    def _set_progress(self, value):
        self.progress["value"] = value

    def _open_path(self, path_text):
        path = plb.Path(path_text)
        if path.exists():
            os.startfile(str(path))
        else:
            messagebox.showwarning("Missing output", "This output file has not been generated yet.")

    def open_comparison_file(self):
        self._open_path(self.comparison_output.get())

    def open_3d_file(self):
        self._open_path(self.output_3d.get())

    def open_comparison_folder(self):
        output_dir = self.pipeline_paths["outputs"] / "rendered_videos" / "default"
        ensure_dirs(output_dir)
        os.startfile(str(output_dir))

    def open_3d_folder(self):
        output_dir = self.pipeline_paths["outputs"] / "rendered_videos_3d" / "default"
        ensure_dirs(output_dir)
        os.startfile(str(output_dir))


if __name__ == "__main__":
    app = FacialAnimationApp()
    app.mainloop()
