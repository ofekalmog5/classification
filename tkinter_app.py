import threading
import os
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List

from backend.app.core import classify, classify_and_export, rasterize_vectors_onto_classification


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


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Material Classification")
        self.root.geometry("920x640")

        self.raster_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.imagery_mode = tk.StringVar(value="regular")
        self.smoothing = tk.StringVar(value="none")
        self.use_spectral = tk.BooleanVar(value=True)
        self.use_texture = tk.BooleanVar(value=True)
        self.use_indices = tk.BooleanVar(value=True)
        self.class_count = tk.IntVar(value=3)
        self.use_tiling = tk.BooleanVar(value=False)
        self.tile_workers = tk.IntVar(value=max(1, os.cpu_count() or 1))
        self.use_max_threads = tk.BooleanVar(value=False)
        self.detect_shadows = tk.BooleanVar(value=False)

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
        ttk.Button(header, text="Browse", command=self._pick_raster).grid(row=0, column=2)

        ttk.Label(header, text="Output file:").grid(row=1, column=0, sticky=tk.W, pady=(8, 0))
        ttk.Entry(header, textvariable=self.output_path, width=60).grid(row=1, column=1, padx=6, pady=(8, 0))
        ttk.Button(header, text="Browse", command=self._pick_output).grid(row=1, column=2, pady=(8, 0))

        ttk.Label(header, text="Post-smoothing:").grid(row=2, column=0, sticky=tk.W, pady=(8, 0))
        smoothing_box = ttk.Combobox(header, textvariable=self.smoothing, values=["none", "median_1", "median_2", "median_3", "median_5"], width=20)
        smoothing_box.grid(row=2, column=1, sticky=tk.W, pady=(8, 0))
        smoothing_box.state(["readonly"])

        perf_frame = ttk.LabelFrame(header, text="Performance", padding=8)
        perf_frame.grid(row=0, column=3, rowspan=3, sticky=tk.NE, padx=(12, 0), pady=(0, 0))
        ttk.Checkbutton(perf_frame, text="Use tile processing", variable=self.use_tiling).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(perf_frame, text="Workers:").grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
        ttk.Spinbox(perf_frame, from_=1, to=64, textvariable=self.tile_workers, width=6).grid(row=1, column=1, padx=(6, 0), pady=(6, 0))
        ttk.Checkbutton(perf_frame, text="Use max threads", variable=self.use_max_threads).grid(row=2, column=0, sticky=tk.W, pady=(6, 0))
        
        class_frame = ttk.LabelFrame(header, text="Classification", padding=8)
        class_frame.grid(row=0, column=4, rowspan=3, sticky=tk.NE, padx=(8, 0), pady=(0, 0))
        ttk.Checkbutton(class_frame, text="Detect shadows", variable=self.detect_shadows).grid(row=0, column=0, sticky=tk.W)

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
        
        self.recommendation_label = ttk.Label(classes_frame, text="", foreground="blue", font=("", 8))
        self.recommendation_label.grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=(4, 0))

        self.class_list = tk.Listbox(classes_frame, height=10)
        self.class_list.grid(row=2, column=0, columnspan=4, sticky=tk.NSEW, pady=(8, 0))

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
        
        self.status = tk.StringVar(value="Idle")
        ttk.Label(footer, textvariable=self.status).pack(side=tk.LEFT)

        self._generate_classes()
        self._apply_mode()

    def _pick_raster(self) -> None:
        path = filedialog.askopenfilename(title="Select raster", filetypes=[("GeoTIFF", "*.tif *.tiff"), ("All", "*.*")])
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

    def _recommend_clusters(self) -> None:
        if not self.raster_path.get():
            messagebox.showwarning("Missing raster", "Select a raster image first.")
            return
        
        self.recommendation_label.config(text="Analyzing image... please wait")
        self.root.update()
        thread = threading.Thread(target=self._run_recommendation, daemon=True)
        thread.start()

    def _run_recommendation(self) -> None:
        try:
            from backend.app.core import recommend_cluster_count
            recommended = recommend_cluster_count(
                self.raster_path.get(),
                {
                    "spectral": self.use_spectral.get(),
                    "texture": self.use_texture.get(),
                    "indices": self.use_indices.get()
                }
            )
            self.root.after(0, lambda: self._finish_recommendation(recommended))
        except Exception as e:
            self.root.after(0, lambda: self._finish_recommendation(None, str(e)))

    def _finish_recommendation(self, recommended: int = None, error: str = None) -> None:
        if error:
            self.recommendation_label.config(text=f"Error: {error}", foreground="red")
        elif recommended:
            self.class_count.set(recommended)
            self.recommendation_label.config(
                text=f"✓ Recommended: {recommended} materials (applied)", 
                foreground="green"
            )
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

    def _run_step1(self) -> None:
        """Run only classification (KMEANS + export)"""
        if not self.raster_path.get():
            messagebox.showwarning("Missing raster", "Select a raster image.")
            return
        self.status.set("Running Step 1: Classification...")
        thread = threading.Thread(target=self._run_step1_job, daemon=True)
        thread.start()

    def _run_step1_job(self) -> None:
        """Execute Step 1: Classification and export"""
        output = self.output_path.get() if self.output_path.get() else None
        max_threads = os.cpu_count() if self.use_max_threads.get() else None
        try:
            result = classify_and_export(
                raster_path=self.raster_path.get(),
                classes=[item.__dict__ for item in self.classes],
                smoothing=self.smoothing.get(),
                feature_flags={
                    "spectral": self.use_spectral.get(),
                    "texture": self.use_texture.get(),
                    "indices": self.use_indices.get()
                },
                output_path=output,
                tile_mode=self.use_tiling.get(),
                tile_max_pixels=512 * 512,
                tile_overlap=0,
                tile_output_dir=None,
                tile_workers=self.tile_workers.get(),
                detect_shadows=self.detect_shadows.get(),
                max_threads=max_threads
            )
            self.root.after(0, lambda: self._finish_step1(result))
        except Exception as e:
            error_result = {"status": "error", "message": str(e)}
            self.root.after(0, lambda: self._finish_step1(error_result))

    def _finish_step1(self, result: Dict[str, object]) -> None:
        if result.get("status") == "ok":
            output_file = result.get('outputPath')
            if self.use_tiling.get() and output_file:
                self.output_path.set(output_file)
            messagebox.showinfo("Step 1 Complete", f"Classification saved:\n{output_file}")
            self.status.set("Step 1 Done")
        else:
            error_msg = result.get("message", str(result))
            messagebox.showerror("Error", f"Step 1 failed:\n{error_msg}")
            self.status.set("Step 1 Error")

    def _run_step2(self) -> None:
        """Run only vector rasterization on existing classification"""
        if not self.output_path.get():
            messagebox.showwarning("Missing input", "Provide the classification file path from Step 1.")
            return
        if self.use_tiling.get() and not Path(self.output_path.get()).is_dir():
            messagebox.showwarning("Missing tiles", "When tile processing is enabled, select the tiles directory from Step 1.")
            return
        if not self.vector_layers:
            messagebox.showwarning("No vectors", "Add at least one vector layer.")
            return
        self.status.set("Running Step 2: Vector rasterization...")
        thread = threading.Thread(target=self._run_step2_job, daemon=True)
        thread.start()

    def _run_step2_job(self) -> None:
        """Execute Step 2: Rasterize vectors onto classification"""
        max_threads = os.cpu_count() if self.use_max_threads.get() else None
        try:
            # For Step 2, the output_path is the input classification file
            classification_file = self.output_path.get()
            if self.use_tiling.get() and Path(classification_file).is_dir():
                output_path = str(Path(classification_file).with_name(Path(classification_file).name + "_with_vectors_tiles"))
            else:
                # Generate output path if not explicitly set
                output_path = str(Path(classification_file).parent / (Path(classification_file).stem + "_with_vectors.tif"))
            
            result = rasterize_vectors_onto_classification(
                classification_path=classification_file,
                vector_layers=[item.__dict__ for item in self.vector_layers],
                classes=[item.__dict__ for item in self.classes],
                output_path=output_path,
                tile_mode=self.use_tiling.get(),
                tile_max_pixels=512 * 512,
                tile_overlap=0,
                tile_output_dir=None,
                tile_workers=self.tile_workers.get(),
                max_threads=max_threads
            )
            self.root.after(0, lambda: self._finish_step2(result))
        except Exception as e:
            error_result = {"status": "error", "message": str(e)}
            self.root.after(0, lambda: self._finish_step2(error_result))

    def _finish_step2(self, result: Dict[str, object]) -> None:
        if result.get("status") == "ok":
            output_file = result.get('outputPath')
            messagebox.showinfo("Step 2 Complete", f"Vector rasterization complete:\n{output_file}")
            self.status.set("Step 2 Done")
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
        thread = threading.Thread(target=self._run_full_job, daemon=True)
        thread.start()

    def _run_full_job(self) -> None:
        """Execute full pipeline"""
        output = self.output_path.get() if self.output_path.get() else None
        max_threads = os.cpu_count() if self.use_max_threads.get() else None
        try:
            result = classify(
                raster_path=self.raster_path.get(),
                classes=[item.__dict__ for item in self.classes],
                vector_layers=[item.__dict__ for item in self.vector_layers],
                smoothing=self.smoothing.get(),
                feature_flags={
                    "spectral": self.use_spectral.get(),
                    "texture": self.use_texture.get(),
                    "indices": self.use_indices.get()
                },
                output_path=output,
                tile_mode=self.use_tiling.get(),
                tile_max_pixels=512 * 512,
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
        if result.get("status") == "ok":
            messagebox.showinfo("Done", f"Saved: {result.get('outputPath')}")
            self.status.set("Done")
        else:
            error_msg = result.get("message", str(result))
            messagebox.showerror("Error", error_msg)
            self.status.set("Error")



def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
