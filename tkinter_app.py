import threading
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List

from backend.app.core import (classify, classify_and_export, rasterize_vectors_onto_classification,
                               MEA_CLASSES as _MEA_CLASSES_CORE, train_kmeans_model,
                               build_shared_color_table, suggest_tile_size,
                               _fmt_duration, _build_stats_table)


@dataclass
class ClassItem:
    id: str
    name: str
    color: str


@dataclass
class VectorLayer:
    id: str
    name: str
    filePath: str
    classId: str
    overrideColor: tuple | None = None


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Material Classification")
        self.root.geometry("1180x780")
        self.root.minsize(1080, 700)

        self.raster_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.imagery_mode = tk.StringVar(value="regular")
        self.use_spectral = tk.BooleanVar(value=True)
        self.use_texture = tk.BooleanVar(value=True)
        self.use_indices = tk.BooleanVar(value=True)
        self.class_count = tk.IntVar(value=3)
        self.use_tiling = tk.BooleanVar(value=False)
        self.tile_size = tk.StringVar(value="Auto")
        self.tile_workers = tk.IntVar(value=max(1, os.cpu_count() or 1))
        self.image_workers = tk.IntVar(value=min(4, max(1, os.cpu_count() or 1)))
        self.use_max_threads = tk.BooleanVar(value=False)
        self.detect_shadows = tk.BooleanVar(value=False)
        self.share_model = tk.BooleanVar(value=True)
        self.export_format = tk.StringVar(value="tif")

        self.classes: List[ClassItem] = []
        self.vector_layers: List[VectorLayer] = []

        self._build_ui()

    def _build_ui(self) -> None:
        header = ttk.Frame(self.root, padding=12)
        header.pack(fill=tk.X)
        header.columnconfigure(1, weight=1)
        header.columnconfigure(3, weight=0)

        ttk.Label(header, text="Input raster:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(header, textvariable=self.raster_path, width=60).grid(row=0, column=1, padx=6)
        ttk.Button(header, text="Browse File", command=self._pick_raster).grid(row=0, column=2)
        ttk.Button(header, text="Browse Folder", command=self._pick_raster_folder).grid(row=0, column=3)

        ttk.Label(header, text="Output file:").grid(row=1, column=0, sticky=tk.W, pady=(8, 0))
        ttk.Entry(header, textvariable=self.output_path, width=60).grid(row=1, column=1, padx=6, pady=(8, 0))
        ttk.Button(header, text="Browse", command=self._pick_output).grid(row=1, column=2, pady=(8, 0))
        ttk.Button(header, text="Output Folder", command=self._pick_output_folder).grid(row=1, column=3, pady=(8, 0))

        perf_frame = ttk.LabelFrame(header, text="Performance", padding=8)
        perf_frame.grid(row=0, column=4, rowspan=3, sticky=tk.NE, padx=(12, 0), pady=(0, 0))
        ttk.Checkbutton(perf_frame, text="Use tile processing", variable=self.use_tiling).grid(row=0, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(perf_frame, text="Tile size:").grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
        tile_size_box = ttk.Combobox(perf_frame, textvariable=self.tile_size, values=["Auto", "256", "512", "1024", "2048", "4096"], width=6)
        tile_size_box.grid(row=1, column=1, padx=(6, 0), pady=(6, 0))
        tile_size_box.state(["readonly"])
        ttk.Label(perf_frame, text="Workers:").grid(row=2, column=0, sticky=tk.W, pady=(6, 0))
        ttk.Spinbox(perf_frame, from_=1, to=64, textvariable=self.tile_workers, width=6).grid(row=2, column=1, padx=(6, 0), pady=(6, 0))
        ttk.Label(perf_frame, text="Image workers:").grid(row=3, column=0, sticky=tk.W, pady=(6, 0))
        ttk.Spinbox(perf_frame, from_=1, to=32, textvariable=self.image_workers, width=6).grid(row=3, column=1, padx=(6, 0), pady=(6, 0))
        ttk.Checkbutton(perf_frame, text="Use max threads", variable=self.use_max_threads).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(6, 0))
        
        class_frame = ttk.LabelFrame(header, text="Classification", padding=8)
        class_frame.grid(row=0, column=5, rowspan=3, sticky=tk.NE, padx=(8, 0), pady=(0, 0))
        ttk.Checkbutton(class_frame, text="Detect shadows", variable=self.detect_shadows).grid(row=0, column=0, sticky=tk.W)
        ttk.Checkbutton(class_frame, text="Share model (batch)", variable=self.share_model).grid(row=1, column=0, sticky=tk.W, pady=(4, 0))
        ttk.Label(class_frame, text="Export format:").grid(row=2, column=0, sticky=tk.W, pady=(6, 0))
        fmt_frame = ttk.Frame(class_frame)
        fmt_frame.grid(row=3, column=0, sticky=tk.W)
        ttk.Radiobutton(fmt_frame, text="TIF", variable=self.export_format, value="tif").pack(side=tk.LEFT)
        ttk.Radiobutton(fmt_frame, text="IMG", variable=self.export_format, value="img").pack(side=tk.LEFT, padx=(8, 0))

        mode_frame = ttk.LabelFrame(header, text="Imagery type", padding=8)
        mode_frame.grid(row=3, column=0, columnspan=3, sticky=tk.W + tk.E, pady=(12, 0))
        ttk.Radiobutton(
            mode_frame,
            text="Regular (RGB)",
            variable=self.imagery_mode,
            value="regular",
            command=self._apply_mode
        ).grid(row=0, column=0, padx=8)
        ttk.Radiobutton(
            mode_frame,
            text="Multispectral",
            variable=self.imagery_mode,
            value="multispectral",
            command=self._apply_mode
        ).grid(row=0, column=1, padx=8)

        features_frame = ttk.LabelFrame(header, text="Features", padding=8)
        features_frame.grid(row=4, column=0, columnspan=3, sticky=tk.W + tk.E, pady=(12, 0))
        ttk.Checkbutton(features_frame, text="Spectral", variable=self.use_spectral).grid(row=0, column=0, padx=8)
        ttk.Checkbutton(features_frame, text="Texture", variable=self.use_texture).grid(row=0, column=1, padx=8)
        self.indices_check = ttk.Checkbutton(features_frame, text="Indices", variable=self.use_indices)
        self.indices_check.grid(row=0, column=2, padx=8)

        body = ttk.Frame(self.root, padding=12)
        body.pack(fill=tk.BOTH, expand=True)

        classes_frame = ttk.LabelFrame(body, text="Materials", padding=10)
        classes_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 8))

        ttk.Label(classes_frame, text="Number of materials:").grid(row=0, column=0, sticky=tk.W)
        ttk.Spinbox(classes_frame, from_=2, to=30, textvariable=self.class_count, width=6).grid(row=0, column=1, padx=6)
        ttk.Button(classes_frame, text="Apply", command=self._generate_classes).grid(row=0, column=2, padx=4)
        ttk.Button(classes_frame, text="Recommend", command=self._recommend_clusters).grid(row=0, column=3, padx=4)
        ttk.Button(classes_frame, text="Quick Recommend", command=self._quick_recommend_clusters).grid(row=0, column=4, padx=4)
        ttk.Button(classes_frame, text="MEA Mode", command=self._set_mea_mode).grid(row=0, column=5, padx=4)
        
        self.recommendation_label = ttk.Label(classes_frame, text="", foreground="blue", font=("", 8))
        self.recommendation_label.grid(row=1, column=0, columnspan=6, sticky=tk.W, pady=(4, 0))

        self.class_list = tk.Listbox(classes_frame, height=10)
        self.class_list.grid(row=2, column=0, columnspan=6, sticky=tk.NSEW, pady=(8, 0))

        vectors_frame = ttk.LabelFrame(body, text="Vector Overlay (Optional)", padding=10)
        vectors_frame.grid(row=0, column=1, sticky=tk.NSEW)
        
        ttk.Label(vectors_frame, text="Vectors will be drawn in yellow on result", font=("", 8)).grid(row=0, column=0, columnspan=5, sticky=tk.W, pady=(0, 4))

        self.vector_class_var = tk.StringVar()
        self.vector_class_menu = ttk.Combobox(vectors_frame, textvariable=self.vector_class_var, state="readonly")
        self.vector_class_menu.grid(row=1, column=0, padx=4)
        ttk.Button(vectors_frame, text="Attach", command=self._add_vector).grid(row=1, column=1, padx=4)
        ttk.Button(vectors_frame, text="Up", command=lambda: self._move_vector(-1)).grid(row=1, column=2, padx=4)
        ttk.Button(vectors_frame, text="Down", command=lambda: self._move_vector(1)).grid(row=1, column=3, padx=4)
        ttk.Button(vectors_frame, text="Remove", command=self._remove_vector).grid(row=1, column=4, padx=4)

        self.vector_list = tk.Listbox(vectors_frame, height=10)
        self.vector_list.grid(row=2, column=0, columnspan=5, sticky=tk.NSEW, pady=(8, 0))

        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        footer = ttk.Frame(self.root, padding=12)
        footer.pack(fill=tk.X)

        button_frame = ttk.Frame(footer)
        button_frame.pack(side=tk.RIGHT)
        
        ttk.Button(button_frame, text="Step 1: Run Classification", command=self._run_step1).pack(side=tk.RIGHT, padx=4)
        ttk.Button(button_frame, text="Step 2: Rasterize Vectors", command=self._run_step2).pack(side=tk.RIGHT, padx=4)
        ttk.Button(button_frame, text="Full Pipeline", command=self._run_full).pack(side=tk.RIGHT, padx=4)
        ttk.Button(button_frame, text="Run for MEA", command=self._run_mea).pack(side=tk.RIGHT, padx=4)
        
        self.status = tk.StringVar(value="Idle")
        ttk.Label(footer, textvariable=self.status).pack(side=tk.LEFT)
        self.progressbar = ttk.Progressbar(footer, mode='indeterminate', length=220)
        self.progressbar.pack(side=tk.LEFT, padx=(12, 0))
        self.progress_text = tk.StringVar(value="")
        ttk.Label(footer, textvariable=self.progress_text, width=14).pack(side=tk.LEFT, padx=(6, 0))

        self._generate_classes()
        self._apply_mode()

    def _get_tile_max_pixels(self, raster_path: str = "") -> int:
        """Return tile_max_pixels for the current tile-size setting.

        When the user selects 'Auto', the ideal tile size is computed from
        the raster dimensions and available system RAM.  Falls back to 1024²
        if the raster cannot be opened or no path is supplied.
        """
        val = self.tile_size.get()
        if val == "Auto":
            if raster_path:
                try:
                    workers = max(1, self.tile_workers.get())
                    side = suggest_tile_size(raster_path, workers=workers)
                    return side * side
                except Exception:
                    pass
            return 2048 * 2048  # safe default when no raster path yet
        try:
            side = int(val)
            return side * side
        except (ValueError, TypeError):
            return 1024 * 1024

    def _pick_raster(self) -> None:
        path = filedialog.askopenfilename(title="Select raster", filetypes=[("GeoTIFF", "*.tif *.tiff"), ("All", "*.*")])
        if path:
            self.raster_path.set(path)

    def _pick_raster_folder(self) -> None:
        path = filedialog.askdirectory(title="Select raster folder")
        if path:
            self.raster_path.set(path)
    
    def _pick_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save output as",
            filetypes=[("GeoTIFF", "*.tif"), ("All", "*.*")],
            defaultextension=".tif"
        )
        if path:
            self.output_path.set(path)

    def _pick_output_folder(self) -> None:
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_path.set(path)

    @staticmethod
    def _resolve_single_output_path(input_file: Path, output_value: str | None, suffix: str) -> str | None:
        if not output_value:
            return None
        out_path = Path(output_value)
        if out_path.is_dir() or not out_path.suffix:
            out_path.mkdir(parents=True, exist_ok=True)
            return str(out_path / f"{input_file.stem}{suffix}.tif")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return str(out_path)

    def _generate_classes(self) -> None:
        count = max(2, int(self.class_count.get() or 2))
        colors = self._palette(count)
        self.classes = []
        self.class_list.delete(0, tk.END)
        for idx in range(count):
            item = ClassItem(
                id=f"class-{idx + 1}",
                name=f"Class {idx + 1}",
                color=colors[idx]
            )
            self.classes.append(item)
            self.class_list.insert(tk.END, f"{item.name} ({item.color})")
        self._refresh_vector_classes()

    def _refresh_vector_classes(self) -> None:
        values = [item.name for item in self.classes]
        self.vector_class_menu["values"] = values
        if values:
            self.vector_class_var.set(values[0])

    def _add_vector(self) -> None:
        if not self.classes:
            messagebox.showwarning("No classes", "Set the number of materials first.")
            return
        path = filedialog.askopenfilename(title="Select vector layer", filetypes=[("Shapefile", "*.shp"), ("All", "*.*")])
        if not path:
            return
        class_name = self.vector_class_var.get()
        class_item = next((item for item in self.classes if item.name == class_name), None)
        if not class_item:
            messagebox.showwarning("Missing class", "Select a class for the vector layer.")
            return
        layer = VectorLayer(
            id=f"vector-{len(self.vector_layers) + 1}",
            name=Path(path).name,
            filePath=path,
            classId=class_item.id
        )
        self.vector_layers.append(layer)
        self._render_vector_list()

    def _render_vector_list(self) -> None:
        self.vector_list.delete(0, tk.END)
        for idx, layer in enumerate(self.vector_layers, start=1):
            class_name = next((item.name for item in self.classes if item.id == layer.classId), "Unknown")
            self.vector_list.insert(tk.END, f"{idx}. {layer.name} -> {class_name}")

    def _move_vector(self, direction: int) -> None:
        selection = self.vector_list.curselection()
        if not selection:
            return
        index = selection[0]
        new_index = index + direction
        if new_index < 0 or new_index >= len(self.vector_layers):
            return
        self.vector_layers[index], self.vector_layers[new_index] = (
            self.vector_layers[new_index],
            self.vector_layers[index]
        )
        self._render_vector_list()
        self.vector_list.selection_set(new_index)

    def _remove_vector(self) -> None:
        selection = self.vector_list.curselection()
        if not selection:
            return
        index = selection[0]
        self.vector_layers.pop(index)
        self._render_vector_list()
    
    def _apply_mode(self) -> None:
        if self.imagery_mode.get() == "regular":
            self.use_indices.set(False)
            self.indices_check.state(["disabled"])
        else:
            self.indices_check.state(["!disabled"])

    # Maximum images to sample in quick-recommend mode
    _QUICK_SAMPLE_COUNT = 5

    # Predefined MEA material classes (sourced from core)
    _MEA_CLASSES = _MEA_CLASSES_CORE

    @property
    def _ext(self) -> str:
        """Current output file extension based on export format selection."""
        return ".img" if self.export_format.get() == "img" else ".tif"

    def _recommend_clusters(self) -> None:
        self._start_recommendation(quick=False)

    def _quick_recommend_clusters(self) -> None:
        self._start_recommendation(quick=True)

    def _start_recommendation(self, quick: bool = False) -> None:
        if not self.raster_path.get():
            messagebox.showwarning("Missing raster", "Select a raster image first.")
            return

        raster_input = Path(self.raster_path.get())
        if raster_input.is_dir():
            label = f"Quick-scanning folder (up to {self._QUICK_SAMPLE_COUNT} images)... please wait" if quick else "Scanning folder... please wait"
        else:
            label = "Analyzing image... please wait"
        self.recommendation_label.config(text=label)
        self.root.update()
        thread = threading.Thread(target=self._run_recommendation, args=(quick,), daemon=True)
        thread.start()

    def _run_recommendation(self, quick: bool = False) -> None:
        try:
            from backend.app.core import recommend_cluster_count
            import statistics

            feature_flags = {
                "spectral": self.use_spectral.get(),
                "texture": self.use_texture.get(),
                "indices": self.use_indices.get()
            }

            raster_input = Path(self.raster_path.get())

            if raster_input.is_dir():
                all_files = sorted(
                    [
                        p for p in raster_input.rglob("*")
                        if p.is_file() and p.suffix.lower() in {".tif", ".tiff", ".jpg", ".jpeg"}
                    ],
                    key=lambda p: str(p).lower()
                )
                if not all_files:
                    self.root.after(0, lambda: self._finish_recommendation(None, "No images found in folder"))
                    return

                if quick and len(all_files) > self._QUICK_SAMPLE_COUNT:
                    import numpy as np
                    indices = np.linspace(0, len(all_files) - 1, self._QUICK_SAMPLE_COUNT, dtype=int)
                    files = [all_files[i] for i in indices]
                    sample_label = f"quick sample ({len(files)}/{len(all_files)} images)"
                else:
                    files = all_files
                    sample_label = f"{len(files)} images"

                results: list[int] = []
                errors: list[str] = []
                for idx, f in enumerate(files):
                    _status = f"Analyzing image {idx + 1}/{len(files)}: {f.name}"
                    self.root.after(0, lambda s=_status: self.recommendation_label.config(text=s))
                    try:
                        n = recommend_cluster_count(str(f), feature_flags)
                        results.append(n)
                    except Exception as per_err:
                        errors.append(f"{f.name}: {per_err}")

                if not results:
                    _err_msg = "All images failed: " + "; ".join(errors[:3])
                    self.root.after(0, lambda: self._finish_recommendation(None, _err_msg))
                    return

                recommended = round(statistics.median(results))
                _detail = f"median of {sample_label}: {results}"
                self.root.after(0, lambda r=recommended, d=_detail: self._finish_recommendation(r, detail=d))
            else:
                recommended = recommend_cluster_count(str(raster_input), feature_flags)
                self.root.after(0, lambda: self._finish_recommendation(recommended))
        except Exception as e:
            _err_msg = str(e)
            self.root.after(0, lambda: self._finish_recommendation(None, _err_msg))

    def _finish_recommendation(self, recommended: int = None, error: str = None, detail: str = None) -> None:
        if error and recommended is None:
            self.recommendation_label.config(text=f"Error: {error}", foreground="red")
        elif recommended:
            self.class_count.set(recommended)
            label_text = f"✓ Recommended: {recommended} materials (applied)"
            if detail:
                label_text += f"  [{detail}]"
            self.recommendation_label.config(text=label_text, foreground="green")
            self._generate_classes()
        else:
            self.recommendation_label.config(text="Could not determine recommendation", foreground="orange")

    @staticmethod
    def _palette(count: int) -> List[str]:
        """Generate highly distinguishable colors."""
        # Base palette with maximum contrast colors
        base = [
            "#FF0000",  # Pure Red
            "#00FF00",  # Pure Green
            "#0000FF",  # Pure Blue
            "#FFFF00",  # Yellow
            "#FF00FF",  # Magenta
            "#00FFFF",  # Cyan
            "#FF8000",  # Orange
            "#8000FF",  # Purple
            "#00FF80",  # Spring Green
            "#FF0080",  # Rose
            "#0080FF",  # Azure
            "#80FF00",  # Lime
            "#FF4500",  # Orange Red
            "#1E90FF",  # Dodger Blue
            "#32CD32",  # Lime Green
            "#FF1493",  # Deep Pink
            "#FFD700",  # Gold
            "#4B0082",  # Indigo
            "#00CED1",  # Dark Turquoise
            "#FF6347",  # Tomato
            "#9400D3",  # Dark Violet
            "#00FA9A",  # Medium Spring Green
            "#FF69B4",  # Hot Pink
            "#1E90FF",  # Dodger Blue
            "#ADFF2F",  # Green Yellow
            "#DC143C",  # Crimson
            "#00BFFF",  # Deep Sky Blue
            "#7FFF00",  # Chartreuse
            "#FF00FF",  # Fuchsia
            "#20B2AA"   # Light Sea Green
        ]
        
        if count <= len(base):
            return base[:count]
        
        # Generate additional colors using HSV spacing
        colors = base[:]
        import colorsys
        while len(colors) < count:
            # Distribute hue evenly, alternate saturation and value
            hue = (len(colors) * 0.618033988749895) % 1.0  # Golden ratio for even distribution
            sat = 0.9 if len(colors) % 2 == 0 else 0.7
            val = 0.95 if (len(colors) // 2) % 2 == 0 else 0.75
            
            r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
            colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
        
        return colors

    def _start_progress(self) -> None:
        self._run_start = time.time()
        self.progressbar.configure(mode='indeterminate', maximum=100, value=0)
        self.progressbar.start(12)
        self.progress_text.set("")

    def _switch_to_determinate(self, total: int) -> None:
        """Called from worker thread via root.after once total file count is known."""
        self.progressbar.stop()
        self.progressbar.configure(mode='determinate', maximum=total, value=0)
        self.progress_text.set(f"0 / {total}")

    def _update_progress(self, done: int, total: int) -> None:
        pct = int(done / total * 100) if total else 100
        self.progressbar['value'] = done
        self.progress_text.set(f"{done} / {total}  ({pct}%)")

    def _stop_progress(self) -> None:
        self.progressbar.stop()
        self.progressbar.configure(mode='determinate', maximum=100, value=100)
        self.progress_text.set("")

    def _elapsed_str(self) -> str:
        secs = time.time() - getattr(self, '_run_start', time.time())
        if secs < 60:
            return f"{secs:.1f}s"
        mins = int(secs // 60)
        return f"{mins}m {secs % 60:.0f}s"

    def _format_mea_mapping(self, mapping) -> str:
        if not mapping:
            return ""
        lines = ["MEA cluster mapping:"]
        for item in mapping:
            cluster = item.get("cluster", "?")
            material = item.get("material", "UNKNOWN")
            color_hex = item.get("colorHex", "#ffffff")
            color_rgb = item.get("colorRGB")
            if isinstance(color_rgb, (list, tuple)) and len(color_rgb) == 3:
                rgb_text = f"RGB({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]})"
            else:
                rgb_text = "RGB(?, ?, ?)"
            lines.append(f"Cluster {cluster} -> {material} ({color_hex}, {rgb_text})")
        return "\n".join(lines)

    def _run_step1(self) -> None:
        """Run only classification (KMEANS + export)"""
        if not self.raster_path.get():
            messagebox.showwarning("Missing raster", "Select a raster image.")
            return
        self.status.set("Running Step 1: Classification...")
        self.root.after(0, self._start_progress)
        thread = threading.Thread(target=self._run_step1_job, daemon=True)
        thread.start()

    def _run_step1_job(self) -> None:
        """Execute Step 1: Classification and export"""
        output = self.output_path.get() if self.output_path.get() else None
        max_threads = os.cpu_count() if self.use_max_threads.get() else None
        try:
            raster_input = Path(self.raster_path.get())
            # If a folder was given, process all supported files recursively
            if raster_input.is_dir():
                saved = []
                errors = []
                files = sorted(
                    [
                        p for p in raster_input.rglob("*")
                        if p.is_file() and p.suffix.lower() in {".tif", ".tiff", ".jpg", ".jpeg"}
                    ],
                    key=lambda p: str(p).lower()
                )

                feature_flags = {
                    "spectral": self.use_spectral.get(),
                    "texture": self.use_texture.get(),
                    "indices": self.use_indices.get()
                }
                classes_payload = [item.__dict__ for item in self.classes]
                max_image_workers = max(1, int(self.image_workers.get() or 1))

                # --- Always train ONE shared model on ALL images in the batch ---
                # This ensures every image and every tile receive identical
                # cluster → material assignments (no inter-image inconsistency).
                _shared_scaler = None
                _shared_kmeans = None
                _shared_color_table = None
                _shared_mea_mapping = None
                _batch_stages: list = []
                if files:
                    n = len(files)
                    self.root.after(0, lambda n=n: self.status.set(
                        f"Training shared model on {n} image(s)..."))
                    _t_train = time.perf_counter()
                    try:
                        _shared_scaler, _shared_kmeans = train_kmeans_model(
                            [str(f) for f in files],
                            classes_payload,
                            feature_flags,
                        )
                        _batch_stages.append(("Shared model training", time.perf_counter() - _t_train))
                        self.root.after(0, lambda: self.status.set("Building shared color table..."))
                        _t_ct = time.perf_counter()
                        _shared_mea_mapping, _shared_color_table = build_shared_color_table(
                            [str(f) for f in files],
                            _shared_scaler,
                            _shared_kmeans,
                            classes_payload,
                            feature_flags,
                        )
                        _batch_stages.append(("Shared color table", time.perf_counter() - _t_ct))
                        self.root.after(0, lambda n=n: self.status.set(
                            f"Applying shared model to {n} images..."))
                    except Exception as _te:
                        print(f"[warn] Shared model training failed: {_te} — falling back to per-image training")
                        _shared_scaler = _shared_kmeans = _shared_color_table = _shared_mea_mapping = None
                        _batch_stages = []

                def _process_one_image(file_path: Path):
                    output_path_obj = Path(output) if output else None
                    ext = self._ext
                    if output_path_obj:
                        # Write all classification outputs into one folder: output/class
                        if output_path_obj.is_dir() or not output_path_obj.suffix:
                            out_dir = output_path_obj / "class"
                        else:
                            out_dir = output_path_obj.parent / "class"
                    else:
                        out_dir = file_path.parent
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = str(out_dir / (file_path.stem + "_classified" + ext))
                    # For tile mode, direct tiles into the same output folder
                    tile_out = str(out_dir) if self.use_tiling.get() else None
                    result = classify_and_export(
                        raster_path=str(file_path),
                        classes=classes_payload,
                        smoothing="none",
                        feature_flags=feature_flags,
                        output_path=out_path,
                        tile_mode=self.use_tiling.get(),
                        tile_max_pixels=self._get_tile_max_pixels(str(file_path)),
                        tile_overlap=0,
                        tile_output_dir=tile_out,
                        tile_workers=self.tile_workers.get(),
                        detect_shadows=self.detect_shadows.get(),
                        max_threads=max_threads,
                        pretrained_scaler=_shared_scaler,
                        pretrained_kmeans=_shared_kmeans,
                        pretrained_color_table=_shared_color_table,
                        pretrained_mea_mapping=_shared_mea_mapping,
                    )
                    if result.get("status") == "ok":
                        return ("ok", result.get("outputPath") or out_path, result.get("meaMapping"))
                    return ("error", str(file_path), result.get("message", str(result)))

                _total_s1 = len(files)
                _done_s1 = [0]
                first_mea_mapping = None
                _t_images = time.perf_counter()
                self.root.after(0, lambda n=_total_s1: self._switch_to_determinate(n))
                with ThreadPoolExecutor(max_workers=max_image_workers) as executor:
                    futures = {executor.submit(_process_one_image, p): p for p in files}
                    for future in as_completed(futures):
                        src = futures[future]
                        try:
                            item = future.result()
                            if item[0] == "ok":
                                saved.append(item[1])
                                if first_mea_mapping is None and len(item) > 2 and item[2]:
                                    first_mea_mapping = item[2]
                            else:
                                errors.append((item[1], item[2]))
                        except Exception as e:
                            errors.append((str(src), str(e)))
                        _done_s1[0] += 1
                        _d, _t = _done_s1[0], _total_s1
                        self.root.after(0, lambda d=_d, t=_t: self._update_progress(d, t))
                _batch_stages.append((f"Classify {len(files)} image(s)", time.perf_counter() - _t_images))
                _batch_total = sum(d for _, d in _batch_stages)
                _batch_table = _build_stats_table(_batch_stages, _batch_total)

                summary = {"status": "ok", "saved": saved, "errors": errors,
                           "meaMapping": first_mea_mapping, "statsTable": _batch_table}
                self.root.after(0, lambda: self._finish_step1(summary))
            else:
                # Single-file mode: output goes into output/class
                ext = self._ext
                if output:
                    out_obj = Path(output)
                    class_dir = (out_obj / "class") if (out_obj.is_dir() or not out_obj.suffix) else (out_obj.parent / "class")
                    class_dir.mkdir(parents=True, exist_ok=True)
                    output_for_single = str(class_dir / f"{raster_input.stem}_classified{ext}")
                    tile_out_single = str(class_dir) if self.use_tiling.get() else None
                else:
                    output_for_single = None
                    tile_out_single = None
                result = classify_and_export(
                    raster_path=str(raster_input),
                    classes=[item.__dict__ for item in self.classes],
                    smoothing="none",
                    feature_flags={
                        "spectral": self.use_spectral.get(),
                        "texture": self.use_texture.get(),
                        "indices": self.use_indices.get()
                    },
                    output_path=output_for_single,
                    tile_mode=self.use_tiling.get(),
                    tile_max_pixels=self._get_tile_max_pixels(str(raster_input)),
                    tile_overlap=0,
                    tile_output_dir=tile_out_single,
                    tile_workers=self.tile_workers.get(),
                    detect_shadows=self.detect_shadows.get(),
                    max_threads=max_threads
                )
                self.root.after(0, lambda: self._finish_step1(result))
        except Exception as e:
            error_result = {"status": "error", "message": str(e)}
            self.root.after(0, lambda: self._finish_step1(error_result))

    def _finish_step1(self, result: Dict[str, object]) -> None:
        self._stop_progress()
        elapsed = self._elapsed_str()
        mea_text = self._format_mea_mapping(result.get("meaMapping"))
        stats_text = result.get("statsTable", "")
        if result.get("status") == "ok":
            # single-file result
            if "outputPath" in result and not ("saved" in result):
                output_file = result.get('outputPath')
                if output_file:
                    self.output_path.set(output_file)
                extra = ""
                if stats_text:
                    extra += f"\n\n{stats_text}"
                if mea_text:
                    extra += f"\n\n{mea_text}"
                messagebox.showinfo("Step 1 Complete", f"Classification saved:\n{output_file}\n\nTime: {elapsed}{extra}")
                self.status.set(f"Step 1 Done  ({elapsed})")
                return

            # batch summary
            saved = result.get("saved", [])
            errors = result.get("errors", [])
            # Update output_path to the class folder so Step 2 can find the files
            if saved:
                first_saved = Path(saved[0])
                self.output_path.set(str(first_saved.parent))
            msg_lines = [f"Processed: {len(saved)} files saved.  Time: {elapsed}"]
            if stats_text:
                msg_lines.append(stats_text)
            if saved:
                msg_lines.append("Saved files:\n" + "\n".join(saved))
            if errors:
                msg_lines.append("Errors:\n" + "\n".join([f"{p}: {m}" for p, m in errors]))
            if mea_text:
                msg_lines.append(mea_text)
            messagebox.showinfo("Step 1 Complete", "\n\n".join(msg_lines))
            self.status.set(f"Step 1 Done  ({elapsed})")
        else:
            error_msg = result.get("message", str(result))
            messagebox.showerror("Error", f"Step 1 failed:\n{error_msg}")
            self.status.set("Step 1 Error")

    def _run_step2(self) -> None:
        """Run only vector rasterization on existing classification"""
        if not self.output_path.get():
            messagebox.showwarning("Missing input", "Provide classification file or output folder from Step 1.")
            return
        if not self.vector_layers:
            messagebox.showwarning("No vectors", "Add at least one vector layer.")
            return
        self.status.set("Running Step 2: Vector rasterization...")
        self.root.after(0, self._start_progress)
        thread = threading.Thread(target=self._run_step2_job, daemon=True)
        thread.start()

    def _run_step2_job(self) -> None:
        """Execute Step 2: Rasterize vectors onto classification"""
        max_threads = os.cpu_count() if self.use_max_threads.get() else None
        try:
            classification_input = Path(self.output_path.get())
            vector_layers_payload = [item.__dict__ for item in self.vector_layers]
            classes_payload = [item.__dict__ for item in self.classes]

            # Single-file mode if the path points to an existing file
            if classification_input.is_file():
                classification_file = str(classification_input)
                # Write into sibling 'vectorized' folder
                parent = classification_input.parent
                vectorized_dir = (parent.parent / "vectorized") if parent.name == "class" else (parent / "vectorized")
                vectorized_dir.mkdir(parents=True, exist_ok=True)
                output_path = str(vectorized_dir / (classification_input.stem + "_with_vectors" + self._ext))

                result = rasterize_vectors_onto_classification(
                    classification_path=classification_file,
                    vector_layers=vector_layers_payload,
                    classes=classes_payload,
                    output_path=output_path,
                    tile_mode=self.use_tiling.get(),
                    tile_max_pixels=self._get_tile_max_pixels(classification_file),
                    tile_overlap=0,
                    tile_output_dir=str(vectorized_dir) if self.use_tiling.get() else None,
                    tile_workers=self.tile_workers.get(),
                    max_threads=max_threads
                )
                self.root.after(0, lambda: self._finish_step2(result))
                return

            # Folder mode:
            # - existing directory => search inside it
            # - non-existing file-like path => search its parent
            # - non-existing folder-like path => search it if exists
            if classification_input.is_dir():
                search_root = classification_input
                # If coming from Step 1's 'class' subfolder, put vectorized as sibling inside output.
                # Otherwise put vectorized inside the given folder itself.
                if classification_input.name == "class":
                    output_root = classification_input.parent / "vectorized"
                else:
                    output_root = classification_input / "vectorized"
            else:
                candidate_dir = classification_input.parent if classification_input.suffix else classification_input
                if not candidate_dir.exists() or not candidate_dir.is_dir():
                    self.root.after(0, lambda: self._finish_step2({"status": "error", "message": f"Output path not found: {classification_input}"}))
                    return
                search_root = candidate_dir
                if candidate_dir.name == "class":
                    output_root = candidate_dir.parent / "vectorized"
                else:
                    output_root = candidate_dir / "vectorized"

            if search_root.is_dir():
                files = sorted(
                    [
                        p for p in search_root.rglob("*")
                        if p.is_file()
                        and p.suffix.lower() in {".tif", ".tiff"}
                        and not p.stem.endswith("_with_vectors")
                    ],
                    key=lambda p: str(p).lower()
                )

                if not files:
                    self.root.after(0, lambda: self._finish_step2({"status": "error", "message": f"No TIF files found in: {search_root}"}))
                    return

                # Export vectorized files into the selected output folder itself
                saved = []
                errors = []
                max_image_workers = max(1, int(self.image_workers.get() or 1))

                def _process_one_classification(file_path: Path):
                    out_dir = output_root
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = str(out_dir / f"{file_path.stem}_with_vectors{self._ext}")
                    result = rasterize_vectors_onto_classification(
                        classification_path=str(file_path),
                        vector_layers=vector_layers_payload,
                        classes=classes_payload,
                        output_path=out_path,
                        tile_mode=self.use_tiling.get(),
                        tile_max_pixels=self._get_tile_max_pixels(str(file_path)),
                        tile_overlap=0,
                        tile_output_dir=str(output_root) if self.use_tiling.get() else None,
                        tile_workers=self.tile_workers.get(),
                        max_threads=max_threads
                    )
                    if result.get("status") == "ok":
                        return ("ok", result.get("outputPath") or out_path)
                    return ("error", str(file_path), result.get("message", str(result)))

                _total_s2 = len(files)
                _done_s2 = [0]
                self.root.after(0, lambda n=_total_s2: self._switch_to_determinate(n))
                with ThreadPoolExecutor(max_workers=max_image_workers) as executor:
                    futures = {executor.submit(_process_one_classification, p): p for p in files}
                    for future in as_completed(futures):
                        src = futures[future]
                        try:
                            item = future.result()
                            if item[0] == "ok":
                                saved.append(item[1])
                            else:
                                errors.append((item[1], item[2]))
                        except Exception as e:
                            errors.append((str(src), str(e)))
                        _done_s2[0] += 1
                        _d, _t = _done_s2[0], _total_s2
                        self.root.after(0, lambda d=_d, t=_t: self._update_progress(d, t))

                self.root.after(0, lambda: self._finish_step2({"status": "ok", "saved": saved, "errors": errors}))
        except Exception as e:
            error_result = {"status": "error", "message": str(e)}
            self.root.after(0, lambda: self._finish_step2(error_result))

    def _finish_step2(self, result: Dict[str, object]) -> None:
        self._stop_progress()
        elapsed = self._elapsed_str()
        stats_text = result.get("statsTable", "")
        if result.get("status") == "ok":
            if "outputPath" in result and not ("saved" in result):
                output_file = result.get('outputPath')
                extra = f"\n\n{stats_text}" if stats_text else ""
                messagebox.showinfo("Step 2 Complete", f"Vector rasterization complete:\n{output_file}\n\nTime: {elapsed}{extra}")
                self.status.set(f"Step 2 Done  ({elapsed})")
                return

            saved = result.get("saved", [])
            errors = result.get("errors", [])
            msg_lines = [f"Processed: {len(saved)} files vectorized.  Time: {elapsed}"]
            if stats_text:
                msg_lines.append(stats_text)
            if saved:
                msg_lines.append("Saved files:\n" + "\n".join(saved))
            if errors:
                msg_lines.append("Errors:\n" + "\n".join([f"{p}: {m}" for p, m in errors]))
            messagebox.showinfo("Step 2 Complete", "\n\n".join(msg_lines))
            self.status.set(f"Step 2 Done  ({elapsed})")
        else:
            error_msg = result.get("message", str(result))
            messagebox.showerror("Error", f"Step 2 failed:\n{error_msg}")
            self.status.set("Step 2 Error")

    def _run_full(self) -> None:
        """Run full pipeline (classification + vectors)"""
        if not self.raster_path.get():
            messagebox.showwarning("Missing raster", "Select a raster image.")
            return
        self.status.set("Running Full Pipeline...")
        self.root.after(0, self._start_progress)
        thread = threading.Thread(target=self._run_full_job, daemon=True)
        thread.start()

    def _run_full_job(self) -> None:
        """Execute full pipeline"""
        output = self.output_path.get() if self.output_path.get() else None
        max_threads = os.cpu_count() if self.use_max_threads.get() else None
        try:
            raster_input = Path(self.raster_path.get())
            if raster_input.is_dir():
                saved = []
                errors = []
                files = sorted(
                    [
                        p for p in raster_input.rglob("*")
                        if p.is_file() and p.suffix.lower() in {".tif", ".tiff", ".jpg", ".jpeg"}
                    ],
                    key=lambda p: str(p).lower()
                )

                feature_flags = {
                    "spectral": self.use_spectral.get(),
                    "texture": self.use_texture.get(),
                    "indices": self.use_indices.get()
                }
                classes_payload = [item.__dict__ for item in self.classes]
                vector_layers_payload = [item.__dict__ for item in self.vector_layers]
                max_image_workers = max(1, int(self.image_workers.get() or 1))

                # --- Shared model for full-pipeline batch ---
                _fp_shared_scaler     = None
                _fp_shared_kmeans     = None
                _fp_shared_color_table = None
                _fp_shared_mea_mapping = None
                _fp_batch_stages: list = []
                if files:
                    n = len(files)
                    self.root.after(0, lambda n=n: self.status.set(
                        f"Training shared model on {n} image(s)..."))
                    _t_train = time.perf_counter()
                    try:
                        _fp_shared_scaler, _fp_shared_kmeans = train_kmeans_model(
                            [str(f) for f in files],
                            classes_payload,
                            feature_flags,
                        )
                        _fp_batch_stages.append(("Shared model training", time.perf_counter() - _t_train))
                        self.root.after(0, lambda: self.status.set("Building shared color table..."))
                        _t_ct = time.perf_counter()
                        _fp_shared_mea_mapping, _fp_shared_color_table = build_shared_color_table(
                            [str(f) for f in files],
                            _fp_shared_scaler,
                            _fp_shared_kmeans,
                            classes_payload,
                            feature_flags,
                        )
                        _fp_batch_stages.append(("Shared color table", time.perf_counter() - _t_ct))
                        self.root.after(0, lambda n=n: self.status.set(
                            f"Applying shared model to {n} images..."))
                    except Exception as _te:
                        print(f"[warn] Full-pipeline shared model failed: {_te} — per-image fallback")
                        _fp_shared_scaler = _fp_shared_kmeans = None
                        _fp_shared_color_table = _fp_shared_mea_mapping = None
                        _fp_batch_stages = []

                def _process_one_image(file_path: Path):
                    output_path_obj = Path(output) if output else None
                    ext = self._ext
                    if output_path_obj and (output_path_obj.is_dir() or not output_path_obj.suffix):
                        relative_parent = file_path.relative_to(raster_input).parent
                        out_dir = output_path_obj / relative_parent
                        out_dir.mkdir(parents=True, exist_ok=True)
                    elif output_path_obj:
                        out_dir = output_path_obj.parent
                        out_dir.mkdir(parents=True, exist_ok=True)
                    else:
                        out_dir = file_path.parent
                    out_path = str(out_dir / (file_path.stem + "_full" + ext))
                    result = classify(
                        raster_path=str(file_path),
                        classes=classes_payload,
                        vector_layers=vector_layers_payload,
                        smoothing="none",
                        feature_flags=feature_flags,
                        output_path=out_path,
                        tile_mode=self.use_tiling.get(),
                        tile_max_pixels=self._get_tile_max_pixels(str(file_path)),
                        tile_overlap=0,
                        tile_output_dir=None,
                        tile_workers=self.tile_workers.get(),
                        detect_shadows=self.detect_shadows.get(),
                        max_threads=max_threads,
                        pretrained_scaler=_fp_shared_scaler,
                        pretrained_kmeans=_fp_shared_kmeans,
                        pretrained_color_table=_fp_shared_color_table,
                        pretrained_mea_mapping=_fp_shared_mea_mapping,
                    )
                    if result.get("status") == "ok":
                        return ("ok", result.get("outputPath") or out_path, result.get("meaMapping"))
                    return ("error", str(file_path), result.get("message", str(result)))

                _total_fl = len(files)
                _done_fl = [0]
                first_mea_mapping = None
                _t_images = time.perf_counter()
                self.root.after(0, lambda n=_total_fl: self._switch_to_determinate(n))
                with ThreadPoolExecutor(max_workers=max_image_workers) as executor:
                    futures = {executor.submit(_process_one_image, p): p for p in files}
                    for future in as_completed(futures):
                        src = futures[future]
                        try:
                            item = future.result()
                            if item[0] == "ok":
                                saved.append(item[1])
                                if first_mea_mapping is None and len(item) > 2 and item[2]:
                                    first_mea_mapping = item[2]
                            else:
                                errors.append((item[1], item[2]))
                        except Exception as e:
                            errors.append((str(src), str(e)))
                        _done_fl[0] += 1
                        _d, _t = _done_fl[0], _total_fl
                        self.root.after(0, lambda d=_d, t=_t: self._update_progress(d, t))
                _fp_batch_stages.append((f"Full pipeline {len(files)} image(s)", time.perf_counter() - _t_images))
                _fp_batch_total = sum(d for _, d in _fp_batch_stages)
                _fp_batch_table = _build_stats_table(_fp_batch_stages, _fp_batch_total)

                summary = {"status": "ok", "saved": saved, "errors": errors,
                           "meaMapping": first_mea_mapping, "statsTable": _fp_batch_table}
                self.root.after(0, lambda: self._finish_full(summary))
            else:
                ext = self._ext
                if output:
                    out_obj = Path(output)
                    if out_obj.is_dir() or not out_obj.suffix:
                        out_obj.mkdir(parents=True, exist_ok=True)
                        output_for_single = str(out_obj / f"{raster_input.stem}_full{ext}")
                    else:
                        out_obj.parent.mkdir(parents=True, exist_ok=True)
                        output_for_single = str(out_obj)
                else:
                    output_for_single = str(raster_input.parent / f"{raster_input.stem}_full{ext}")
                result = classify(
                    raster_path=self.raster_path.get(),
                    classes=[item.__dict__ for item in self.classes],
                    vector_layers=[item.__dict__ for item in self.vector_layers],
                    smoothing="none",
                    feature_flags={
                        "spectral": self.use_spectral.get(),
                        "texture": self.use_texture.get(),
                        "indices": self.use_indices.get()
                    },
                    output_path=output_for_single,
                    tile_mode=self.use_tiling.get(),
                    tile_max_pixels=self._get_tile_max_pixels(self.raster_path.get()),
                    tile_overlap=0,
                    tile_output_dir=None,
                    tile_workers=self.tile_workers.get(),
                    detect_shadows=self.detect_shadows.get(),
                    max_threads=max_threads
                )
                self.root.after(0, lambda: self._finish_full(result))
        except Exception as e:
            error_result = {"status": "error", "message": str(e)}
            self.root.after(0, lambda: self._finish_full(error_result))

    def _finish_full(self, result: Dict[str, object]) -> None:
        self._stop_progress()
        elapsed = self._elapsed_str()
        mea_text = self._format_mea_mapping(result.get("meaMapping"))
        stats_text = result.get("statsTable", "")
        if result.get("status") == "ok":
            # single-file result
            if "outputPath" in result and not ("saved" in result):
                extra = ""
                if stats_text:
                    extra += f"\n\n{stats_text}"
                if mea_text:
                    extra += f"\n\n{mea_text}"
                messagebox.showinfo("Done", f"Saved: {result.get('outputPath')}\n\nTime: {elapsed}{extra}")
                self.status.set(f"Done  ({elapsed})")
                return

            saved = result.get("saved", [])
            errors = result.get("errors", [])
            msg_lines = [f"Processed: {len(saved)} files saved.  Time: {elapsed}"]
            if stats_text:
                msg_lines.append(stats_text)
            if saved:
                msg_lines.append("Saved files:\n" + "\n".join(saved))
            if errors:
                msg_lines.append("Errors:\n" + "\n".join([f"{p}: {m}" for p, m in errors]))
            if mea_text:
                msg_lines.append(mea_text)
            messagebox.showinfo("Done", "\n\n".join(msg_lines))
            self.status.set(f"Done  ({elapsed})")
        else:
            error_msg = result.get("message", str(result))
            messagebox.showerror("Error", error_msg)
            self.status.set("Error")

    # ── MEA Pipeline ─────────────────────────────────────────────────────────

    def _set_mea_mode(self) -> None:
        """Load MEA predefined classes into the UI without running anything."""
        self.classes = [
            ClassItem(id=c["id"], name=c["name"], color=c["color"])
            for c in self._MEA_CLASSES
        ]
        self.class_count.set(len(self.classes))
        self.class_list.delete(0, tk.END)
        for item in self.classes:
            self.class_list.insert(tk.END, f"{item.name} ({item.color})")
        self._refresh_vector_classes()
        self.recommendation_label.config(
            text=f"✓ MEA mode: {len(self.classes)} materials loaded",
            foreground="green"
        )

    def _run_mea(self) -> None:
        """Load MEA predefined classes and run full pipeline (classification + rasterize if vectors are attached)."""
        if not self.raster_path.get():
            messagebox.showwarning("Missing raster", "Select a raster image or folder first.")
            return

        self._set_mea_mode()

        if self.vector_layers:
            # Resolve each vector layer to a MEA class and inject its RGB as overrideColor
            mea_by_id = {c["id"]: c for c in self._MEA_CLASSES}
            for i, layer in enumerate(self.vector_layers):
                mea_cls = mea_by_id.get(layer.classId)
                if mea_cls is None:
                    mea_cls = self._MEA_CLASSES[i % len(self._MEA_CLASSES)]
                    layer.classId = mea_cls["id"]
                hex_c = mea_cls["color"].lstrip("#")
                layer.overrideColor = (int(hex_c[0:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16))
            self._render_vector_list()
            self.status.set("Running MEA Classification + Rasterize...")
            thread = threading.Thread(target=self._run_full_job, daemon=True)
        else:
            self.status.set("Running MEA Classification...")
            thread = threading.Thread(target=self._run_step1_job, daemon=True)

        self.root.after(0, self._start_progress)
        thread.daemon = True
        thread.start()



def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    App(root)
    root.mainloop()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
