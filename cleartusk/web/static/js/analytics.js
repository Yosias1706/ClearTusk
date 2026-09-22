/* Analytics dashboard charts. */
(() => {
  "use strict";

  const node = document.getElementById("dashboard-data");
  const charts = window.ClearTuskCharts;
  if (!node || !charts) return;

  let data;
  try {
    data = JSON.parse(node.textContent);
  } catch (_) {
    return;
  }

  // Quality by call profile — two measures, both percentages, one axis.
  const byType = data.quality_by_call_type || [];
  if (byType.length) {
    charts.legend("legend-call-type", [{ label: "Call retained" }, { label: "Machine removed" }]);
    charts.bars(
      "chart-call-type",
      byType.map((row) => row.display_name),
      [
        { label: "Call retained", data: byType.map((row) => row.retention) },
        { label: "Machine removed", data: byType.map((row) => row.suppression) },
      ],
      { percent: true, yTitle: "% of band energy" }
    );
  }

  // Preset selection — one series, so no legend box.
  const presets = Object.entries(data.preset_distribution || {});
  if (presets.length) {
    charts.bars(
      "chart-presets",
      presets.map(([name]) => name),
      [{ label: "Clips", data: presets.map(([, count]) => count) }],
      { horizontal: true }
    );
  }

  const distributions = data.quality_distributions || {};
  if (distributions.retention) {
    charts.bars("chart-retention", distributions.retention.labels, [
      { label: "Clips", data: distributions.retention.counts },
    ], { yTitle: "clips" });
  }
  if (distributions.gain_db) {
    charts.bars("chart-gain", distributions.gain_db.labels, [
      { label: "Clips", data: distributions.gain_db.counts },
    ], { yTitle: "clips" });
  }

  // Detector: four scores per fold, all on a 0–1 scale.
  const folds = (data.detector && data.detector.fold_scores) || [];
  if (folds.length) {
    const metrics = [
      ["accuracy", "Accuracy"],
      ["precision", "Precision"],
      ["recall", "Recall"],
      ["f1", "F1"],
    ];
    charts.legend("legend-folds", metrics.map(([, label]) => ({ label })));
    charts.bars(
      "chart-folds",
      folds.map((fold) => `Fold ${fold.fold}`),
      metrics.map(([key, label]) => ({
        label,
        data: folds.map((fold) => Number((fold[key] ?? 0).toFixed(3))),
      })),
      { yTitle: "score" }
    );
  }

  // Benchmark history — both series are percentages; gain (dB) is deliberately
  // left out rather than given a second y-axis.
  const history = data.benchmark_history || [];
  if (history.length) {
    charts.legend("legend-history", [{ label: "Call retained" }, { label: "Machine removed" }]);
    charts.lines(
      "chart-history",
      history.map((row) => row.label),
      [
        { label: "Call retained", data: history.map((row) => row.retention) },
        { label: "Machine removed", data: history.map((row) => row.suppression) },
      ],
      { percent: true, yTitle: "% of band energy" }
    );
  }

  const corpus = (data.corpus && data.corpus.by_noise_source) || [];
  if (corpus.length) {
    charts.bars(
      "chart-corpus",
      corpus.map((row) => row.noise_source.charAt(0).toUpperCase() + row.noise_source.slice(1)),
      [{ label: "Annotated calls", data: corpus.map((row) => row.clips) }],
      { yTitle: "annotated calls" }
    );
  }

  const throughput = data.throughput || {};
  if (throughput.labels && throughput.labels.length) {
    // Daily counts are discrete events: columns, never a smoothed line.
    charts.bars(
      "chart-activity",
      throughput.labels.map((day) => day.slice(5)),
      [{ label: "Runs", data: throughput.counts }],
      { yTitle: "runs" }
    );
  }
})();
