"use client";

import { useCallback, useEffect, useRef, useState } from "react";

// ─── Types ───────────────────────────────────────────────────────────────────

interface QueueItem {
  id: string;
  name: string;
  status: "pending" | "processing" | "done" | "error";
  original?: string;
  processed?: string;
  mask?: string;
}

type ViewMode = "compare" | "original" | "mask";

// ─── API ─────────────────────────────────────────────────────────────────────

const API = "http://localhost:8000";

async function uploadFile(
  file: File,
  denoise: number
): Promise<{ job_id: string; original: string; processed: string; mask: string }> {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("denoise", denoise.toString());
  const res = await fetch(`${API}/api/process`, { method: "POST", body: fd });
  if (!res.ok) throw new Error("Upload failed");
  return res.json();
}

// ─── Page ────────────────────────────────────────────────────────────────────

export default function Darkroom() {
  const [queue, setQueue] = useState<QueueItem[]>([]);
  const [active, setActive] = useState<string | null>(null);
  const [denoise, setDenoise] = useState(0.3);
  const [viewMode, setViewMode] = useState<ViewMode>("compare");
  const [sliderPos, setSliderPos] = useState(50);
  const [dragging, setDragging] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const canvasRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const activeItem = queue.find((q) => q.id === active);

  // ── Hotkeys ──────────────────────────────────────────────────────────────

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.code === "Space") {
        e.preventDefault();
        setViewMode((v) => (v === "original" ? "compare" : "original"));
      }
      if (e.code === "KeyM") {
        setViewMode((v) => (v === "mask" ? "compare" : "mask"));
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // ── Upload handler ───────────────────────────────────────────────────────

  const handleFiles = useCallback(
    async (files: FileList | null) => {
      if (!files) return;
      for (const file of Array.from(files)) {
        const id = crypto.randomUUID();
        const item: QueueItem = { id, name: file.name, status: "pending" };
        setQueue((q) => [...q, item]);
        setActive(id);

        setQueue((q) =>
          q.map((i) => (i.id === id ? { ...i, status: "processing" } : i))
        );

        try {
          const result = await uploadFile(file, denoise);
          setQueue((q) =>
            q.map((i) =>
              i.id === id
                ? {
                    ...i,
                    status: "done",
                    original: `${API}${result.original}`,
                    processed: `${API}${result.processed}`,
                    mask: `${API}${result.mask}`,
                  }
                : i
            )
          );
        } catch {
          setQueue((q) =>
            q.map((i) => (i.id === id ? { ...i, status: "error" } : i))
          );
        }
      }
    },
    [denoise]
  );

  // ── Drag & Drop ──────────────────────────────────────────────────────────

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      handleFiles(e.dataTransfer.files);
    },
    [handleFiles]
  );

  // ── Slider interaction ───────────────────────────────────────────────────

  const onSliderMove = useCallback(
    (e: React.MouseEvent | MouseEvent) => {
      if (!dragging || !canvasRef.current) return;
      const rect = canvasRef.current.getBoundingClientRect();
      const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
      setSliderPos((x / rect.width) * 100);
    },
    [dragging]
  );

  useEffect(() => {
    if (!dragging) return;
    const up = () => setDragging(false);
    const move = (e: MouseEvent) => onSliderMove(e);
    window.addEventListener("mouseup", up);
    window.addEventListener("mousemove", move);
    return () => {
      window.removeEventListener("mouseup", up);
      window.removeEventListener("mousemove", move);
    };
  }, [dragging, onSliderMove]);

  // ── Status dot ───────────────────────────────────────────────────────────

  function statusColor(s: QueueItem["status"]) {
    if (s === "done") return "bg-emerald-500";
    if (s === "processing") return "bg-amber-500 animate-pulse";
    if (s === "error") return "bg-red-500";
    return "bg-zinc-600";
  }

  // ── Render ───────────────────────────────────────────────────────────────

  return (
    <div className="flex h-screen w-screen overflow-hidden select-none">
      {/* ─── LEFT SIDEBAR: Queue ─────────────────────────────────────── */}
      <aside className="w-[15%] min-w-[180px] border-r border-zinc-800 flex flex-col bg-[#0a0a0a]">
        <div className="px-3 py-4 border-b border-zinc-800">
          <h1 className="text-xs font-bold tracking-[0.2em] uppercase text-zinc-400">
            APS Darkroom
          </h1>
        </div>

        <div className="flex-1 overflow-y-auto">
          {queue.length === 0 && (
            <p className="text-[10px] text-zinc-600 px-3 py-6 text-center">
              Drop images to begin
            </p>
          )}
          {queue.map((item) => (
            <button
              key={item.id}
              onClick={() => setActive(item.id)}
              className={`w-full text-left px-3 py-2 border-b border-zinc-800/50 flex items-center gap-2 transition-colors hover:bg-zinc-900 ${
                active === item.id ? "bg-zinc-900" : ""
              }`}
            >
              <span
                className={`w-1.5 h-1.5 shrink-0 ${statusColor(item.status)}`}
              />
              <span className="text-[11px] text-zinc-300 truncate">
                {item.name}
              </span>
            </button>
          ))}
        </div>

        {/* Hotkey legend */}
        <div className="border-t border-zinc-800 px-3 py-3 text-[9px] text-zinc-600 space-y-1">
          <div>
            <kbd className="text-zinc-500">Space</kbd> Toggle original
          </div>
          <div>
            <kbd className="text-zinc-500">M</kbd> Toggle mask
          </div>
        </div>
      </aside>

      {/* ─── RIGHT CANVAS ────────────────────────────────────────────── */}
      <main
        className="flex-1 relative bg-[#0a0a0a] flex items-center justify-center"
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
      >
        {/* Drag overlay */}
        {dragOver && (
          <div className="absolute inset-0 z-50 bg-zinc-900/80 border-2 border-dashed border-zinc-600 flex items-center justify-center">
            <p className="text-sm text-zinc-400 tracking-wide">
              Drop to process
            </p>
          </div>
        )}

        {/* Empty state */}
        {!activeItem && (
          <div className="text-center">
            <p className="text-zinc-600 text-sm mb-4">
              Drag images here or click to upload
            </p>
            <button
              onClick={() => fileRef.current?.click()}
              className="border border-zinc-700 px-4 py-2 text-xs text-zinc-400 hover:bg-zinc-900 transition-colors"
            >
              Select Files
            </button>
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              multiple
              className="hidden"
              onChange={(e) => handleFiles(e.target.files)}
            />
          </div>
        )}

        {/* Active image — comparison slider */}
        {activeItem && activeItem.status === "done" && (
          <div
            ref={canvasRef}
            className="relative w-full h-full overflow-hidden cursor-col-resize"
            onMouseDown={(e) => {
              setDragging(true);
              onSliderMove(e);
            }}
          >
            {viewMode === "original" && (
              <img
                src={activeItem.original}
                alt="Original"
                className="absolute inset-0 w-full h-full object-contain"
              />
            )}

            {viewMode === "mask" && (
              <img
                src={activeItem.mask}
                alt="Mask"
                className="absolute inset-0 w-full h-full object-contain"
              />
            )}

            {viewMode === "compare" && (
              <>
                {/* After (full) */}
                <img
                  src={activeItem.processed}
                  alt="Processed"
                  className="absolute inset-0 w-full h-full object-contain"
                />
                {/* Before (clipped) */}
                <div
                  className="absolute inset-0 overflow-hidden"
                  style={{ width: `${sliderPos}%` }}
                >
                  <img
                    src={activeItem.original}
                    alt="Original"
                    className="absolute inset-0 w-full h-full object-contain"
                    style={{
                      width: canvasRef.current
                        ? `${canvasRef.current.offsetWidth}px`
                        : "100vw",
                      maxWidth: "none",
                    }}
                  />
                </div>
                {/* Slider line */}
                <div
                  className="absolute top-0 bottom-0 w-px bg-zinc-400 z-10 pointer-events-none"
                  style={{ left: `${sliderPos}%` }}
                >
                  <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-3 h-8 border border-zinc-400 bg-zinc-900 flex items-center justify-center">
                    <div className="w-px h-4 bg-zinc-500" />
                  </div>
                </div>
                {/* Labels */}
                <div className="absolute top-3 left-3 text-[10px] uppercase tracking-widest text-zinc-500 bg-zinc-900/80 px-2 py-1 z-10">
                  Original
                </div>
                <div className="absolute top-3 right-3 text-[10px] uppercase tracking-widest text-zinc-500 bg-zinc-900/80 px-2 py-1 z-10">
                  Processed
                </div>
              </>
            )}

            {/* View mode indicator */}
            {viewMode !== "compare" && (
              <div className="absolute top-3 left-1/2 -translate-x-1/2 text-[10px] uppercase tracking-widest text-zinc-500 bg-zinc-900/80 px-2 py-1 z-10">
                {viewMode}
              </div>
            )}
          </div>
        )}

        {/* Processing state */}
        {activeItem && activeItem.status === "processing" && (
          <div className="text-center">
            <div className="w-6 h-6 border border-zinc-600 border-t-zinc-300 animate-spin mx-auto mb-3" />
            <p className="text-[11px] text-zinc-500">Processing...</p>
          </div>
        )}

        {/* Error state */}
        {activeItem && activeItem.status === "error" && (
          <p className="text-[11px] text-red-500">Pipeline error</p>
        )}

        {/* ─── Right-edge vertical Denoise slider ──────────────────── */}
        <div className="absolute top-1/2 -translate-y-1/2 right-0 z-20 flex flex-col items-center gap-2 bg-zinc-900/80 border-l border-zinc-800 px-2 py-4">
          <span className="text-[11px] text-zinc-500 tabular-nums">
            {denoise.toFixed(2)}
          </span>
          <input
            type="range"
            min={0.25}
            max={0.35}
            step={0.01}
            value={denoise}
            onChange={(e) => setDenoise(parseFloat(e.target.value))}
            className="h-48 accent-zinc-400"
            style={{
              writingMode: "vertical-lr",
              direction: "rtl",
              WebkitAppearance: "slider-vertical",
              width: "20px",
            }}
          />
          <label className="text-[9px] uppercase tracking-[0.15em] text-zinc-600 whitespace-nowrap"
            style={{ writingMode: "vertical-lr", textOrientation: "mixed" }}
          >
            Denoise
          </label>
        </div>
      </main>
    </div>
  );
}
