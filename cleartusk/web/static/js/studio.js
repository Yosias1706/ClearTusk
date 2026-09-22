/* Studio: upload, progress, and result rendering. */
(() => {
  "use strict";

  const form = document.getElementById("upload-form");
  if (!form) return;

  const input = document.getElementById("audio_file");
  const dropzone = document.getElementById("dropzone");
  const fileName = document.getElementById("file-name");
  const submit = document.getElementById("submit-btn");
  const status = document.getElementById("status");
  const progress = document.getElementById("progress");
  const progressBar = document.getElementById("progress-bar");
  const results = document.getElementById("results");
  const placeholder = document.getElementById("placeholder");

  const METRICS = [
    { key: "target_retention_pct", label: "Call retained", suffix: "%", meter: true, hint: (m) => `${Math.round(m.target_band_low_hz)}–${Math.round(m.target_band_high_hz)} Hz protected` },
    { key: "machine_suppression_pct", label: "Machine removed", suffix: "%", meter: true, hint: (m) => `${Math.round(m.machine_band_low_hz)}–${Math.round(m.machine_band_high_hz)} Hz noise band` },
    { key: "target_machine_gain_db", label: "Call-to-noise gain", prefix: "+", suffix: " dB", digits: 2, hint: () => "Band-ratio improvement" },
    { key: "spectral_fidelity", label: "Spectral fidelity", digits: 3, hint: () => "1.0 = in-band spectrum untouched" },
    { key: "peak_frequency_hz", label: "Peak frequency", suffix: " Hz", digits: 1, hint: (m) => (m.peak_in_target_band ? "Inside the protected band" : "Outside the protected band") },
    { key: "core_rumble_retention_pct", label: "Core rumble kept", suffix: "%", digits: 1, hint: () => "10–150 Hz, profile independent" },
  ];

  function setStatus(message, kind) {
    status.textContent = message;
    status.style.color = kind === "error" ? "var(--critical)" : "var(--ink-muted)";
  }

  function showFile(file) {
    if (!file) return;
    fileName.hidden = false;
    fileName.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(1)} MB`;
  }

  dropzone.addEventListener("click", () => input.click());
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      input.click();
    }
  });
  ["dragenter", "dragover"].forEach((type) =>
    dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      dropzone.classList.add("is-dragging");
    })
  );
  ["dragleave", "drop"].forEach((type) =>
    dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      dropzone.classList.remove("is-dragging");
    })
  );
  dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer?.files?.[0];
    if (!file) return;
    const transfer = new DataTransfer();
    transfer.items.add(file);
    input.files = transfer.files;
    showFile(file);
  });
  input.addEventListener("change", () => showFile(input.files[0]));

  function formatMetric(metrics, spec) {
    const value = metrics[spec.key];
    if (value === undefined || value === null) return "—";
    const digits = spec.digits ?? 1;
    return `${spec.prefix || ""}${Number(value).toFixed(digits)}${spec.suffix || ""}`;
  }

  function renderMetrics(metrics) {
    document.getElementById("metric-grid").innerHTML = METRICS.map((spec) => {
      const meter = spec.meter
        ? `<div class="meter"><div class="meter__fill" style="width:${Math.max(0, Math.min(100, metrics[spec.key]))}%"></div></div>`
        : "";
      return `<div class="metric">
          <div class="metric__label">${spec.label}</div>
          <div class="metric__value">${formatMetric(metrics, spec)}</div>
          ${meter}
          <div class="metric__hint">${spec.hint(metrics)}</div>
        </div>`;
    }).join("");
  }

  function renderDetections(detection, audioSeconds) {
    const badge = document.getElementById("detection-badge");
    const summary = document.getElementById("detection-summary");
    const rows = document.getElementById("detection-rows");

    badge.textContent = detection.call_present
      ? `${detection.event_count} event${detection.event_count === 1 ? "" : "s"}`
      : "no calls found";
    summary.textContent = detection.call_present
      ? `Peak confidence ${(detection.peak_confidence * 100).toFixed(1)}% · ${detection.coverage_pct.toFixed(0)}% of the recording · dominant profile ${detection.dominant_call_type}`
      : "No window crossed the detection threshold; the profile came from the spectral balance instead.";

    rows.innerHTML = detection.events.length
      ? detection.events
          .map(
            (event, index) => `<tr>
              <td>${index + 1}</td>
              <td>${event.start_seconds.toFixed(2)} s</td>
              <td>${event.end_seconds.toFixed(2)} s</td>
              <td class="num">${event.duration_seconds.toFixed(2)} s</td>
              <td class="num">${(event.confidence * 100).toFixed(1)}%</td>
              <td>${event.peak_frequency_hz.toFixed(1)} Hz</td>
              <td><span class="badge badge-accent">${event.predicted_call_type}</span></td>
            </tr>`
          )
          .join("")
      : `<tr><td colspan="7" class="text-muted">No events above the threshold.</td></tr>`;

    const charts = window.ClearTuskCharts;
    if (!charts || !detection.timeline.length) return;
    charts.register("detection-chart", (colors) => ({
      type: "line",
      data: {
        labels: detection.timeline.map((point) => point.t.toFixed(1)),
        datasets: [
          {
            label: "Call probability",
            data: detection.timeline.map((point) => point.p),
            borderColor: colors.series[0],
            backgroundColor: `color-mix(in srgb, ${colors.series[0]} 12%, transparent)`,
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 5,
            tension: 0.25,
            fill: true,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: colors.surface,
            titleColor: colors.ink,
            bodyColor: colors.ink,
            borderColor: colors.grid,
            borderWidth: 1,
            callbacks: {
              title: (items) => `${items[0].label} s`,
              label: (item) => ` ${(item.parsed.y * 100).toFixed(1)}% confidence`,
            },
          },
        },
        scales: {
          x: {
            title: { display: true, text: "Window start (s)", color: colors.inkMuted, font: { size: 11 } },
            grid: { display: false },
            border: { color: colors.grid },
            ticks: { color: colors.inkMuted, maxTicksLimit: 10, font: { size: 11 } },
          },
          y: {
            min: 0,
            max: 1,
            grid: { color: colors.grid, drawTicks: false },
            border: { display: false },
            ticks: { color: colors.inkMuted, font: { size: 11 }, callback: (value) => `${value * 100}%` },
          },
        },
      },
    }));
  }

  function renderResult(payload) {
    document.getElementById("result-filename").textContent = payload.filename;
    document.getElementById("result-meta").textContent =
      `${payload.call_type} profile (${payload.call_type_source}) · preset ${payload.preset} · ` +
      `best of ${payload.candidates_evaluated} candidates · ${payload.duration_ms} ms · ${payload.realtime_factor}× real time`;

    document.getElementById("download").href = payload.assets.cleaned_audio;
    document.getElementById("permalink").href = payload.links.self;
    document.getElementById("audio-original").src = payload.assets.original_audio;
    document.getElementById("audio-cleaned").src = payload.assets.cleaned_audio;
    document.getElementById("spectrogram-comparison").src = payload.assets.comparison_spectrogram;

    renderMetrics(payload.metrics);
    renderDetections(payload.detection, payload.audio_seconds);

    placeholder.hidden = true;
    results.hidden = false;
    results.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function upload(formData) {
    return new Promise((resolve, reject) => {
      const request = new XMLHttpRequest();
      request.open("POST", "/api/v1/process");
      request.upload.addEventListener("progress", (event) => {
        if (!event.lengthComputable) return;
        const percent = (event.loaded / event.total) * 70;
        progressBar.style.width = `${percent}%`;
        if (percent >= 69) setStatus("Detecting calls and running the cleaner…");
      });
      request.addEventListener("load", () => {
        progressBar.style.width = "100%";
        let payload;
        try {
          payload = JSON.parse(request.responseText);
        } catch (_) {
          reject(new Error("The server returned an unreadable response."));
          return;
        }
        if (request.status >= 400 || !payload.success) {
          reject(new Error(payload.error || "Processing failed."));
          return;
        }
        resolve(payload);
      });
      request.addEventListener("error", () => reject(new Error("Network error while uploading.")));
      request.send(formData);
    });
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!input.files.length) {
      setStatus("Choose an audio file first.", "error");
      return;
    }

    submit.disabled = true;
    submit.textContent = "Working…";
    progress.hidden = false;
    progressBar.style.width = "0%";
    setStatus("Uploading…");

    try {
      const payload = await upload(new FormData(form));
      renderResult(payload);
      setStatus(`Done in ${payload.duration_ms} ms.`);
    } catch (error) {
      setStatus(error.message, "error");
    } finally {
      submit.disabled = false;
      submit.textContent = "Detect & clean";
      window.setTimeout(() => {
        progress.hidden = true;
        progressBar.style.width = "0%";
      }, 600);
    }
  });
})();
