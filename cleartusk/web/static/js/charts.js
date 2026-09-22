/* Chart helpers shared by the studio and the analytics dashboard.
   Colours come from CSS custom properties, so every chart follows the active
   theme, and all charts re-render when the theme changes. */
(() => {
  "use strict";

  const registry = new Map();

  function token(name, fallback) {
    const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
  }

  function palette() {
    return {
      series: [token("--series-1", "#2a78d6"), token("--series-2", "#eb6834"), token("--series-3", "#1baf7a"), token("--series-4", "#eda100")],
      ink: token("--ink", "#1b1813"),
      inkMuted: token("--ink-muted", "#7d7464"),
      grid: token("--grid", "#e6e0d2"),
      surface: token("--surface", "#ffffff"),
      accent: token("--accent", "#1f5d38"),
      amber: token("--amber", "#b9791a"),
    };
  }

  function baseOptions(colors, { horizontal = false, stacked = false, yTitle = "", percent = false } = {}) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: horizontal ? "y" : "x",
      layout: { padding: { top: 8, right: 8 } },
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: colors.surface,
          titleColor: colors.ink,
          bodyColor: colors.ink,
          borderColor: colors.grid,
          borderWidth: 1,
          padding: 10,
          displayColors: true,
          boxWidth: 10,
          boxHeight: 10,
          usePointStyle: true,
          callbacks: percent
            ? { label: (ctx) => ` ${ctx.dataset.label || ""} ${ctx.parsed[horizontal ? "x" : "y"]}%`.trim() }
            : undefined,
        },
      },
      scales: {
        x: {
          stacked,
          grid: { display: horizontal, color: colors.grid, drawTicks: false },
          border: { color: colors.grid },
          ticks: { color: colors.inkMuted, font: { size: 11 } },
        },
        y: {
          stacked,
          beginAtZero: true,
          grid: { display: !horizontal, color: colors.grid, drawTicks: false },
          border: { display: false },
          ticks: { color: colors.inkMuted, font: { size: 11 }, maxTicksLimit: 6 },
          title: yTitle ? { display: true, text: yTitle, color: colors.inkMuted, font: { size: 11 } } : undefined,
        },
      },
    };
  }

  const barStyle = {
    borderRadius: 4,
    borderSkipped: "start",
    maxBarThickness: 24,
    borderWidth: 2, // drawn in the surface colour: the 2px gap between adjacent bars
    categoryPercentage: 0.7,
    barPercentage: 0.88,
  };

  function bars(canvasId, labels, datasets, options = {}) {
    return register(canvasId, (colors) => ({
      type: "bar",
      data: {
        labels,
        datasets: datasets.map((dataset, index) => ({
          ...barStyle,
          borderColor: colors.surface, // 2px surface gap between adjacent bars
          backgroundColor: dataset.color ? dataset.color(colors) : colors.series[index % colors.series.length],
          ...dataset,
        })),
      },
      options: baseOptions(colors, options),
    }));
  }

  function lines(canvasId, labels, datasets, options = {}) {
    return register(canvasId, (colors) => ({
      type: "line",
      data: {
        labels,
        datasets: datasets.map((dataset, index) => {
          const color = dataset.color ? dataset.color(colors) : colors.series[index % colors.series.length];
          return {
            borderColor: color,
            backgroundColor: color,
            borderWidth: 2,
            pointRadius: 4,
            pointHoverRadius: 6,
            pointBorderColor: colors.surface, // surface ring keeps markers legible on crossings
            pointBorderWidth: 2,
            tension: 0.25,
            fill: false,
            ...dataset,
          };
        }),
      },
      options: baseOptions(colors, options),
    }));
  }

  function area(canvasId, labels, dataset, options = {}) {
    return register(canvasId, (colors) => {
      const color = dataset.color ? dataset.color(colors) : colors.series[0];
      return {
        type: "line",
        data: {
          labels,
          datasets: [
            {
              borderColor: color,
              backgroundColor: `color-mix(in srgb, ${color} 12%, transparent)`,
              borderWidth: 2,
              pointRadius: 0,
              pointHoverRadius: 5,
              pointBorderColor: colors.surface,
              pointBorderWidth: 2,
              tension: 0.3,
              fill: true,
              ...dataset,
            },
          ],
        },
        options: baseOptions(colors, options),
      };
    });
  }

  function register(canvasId, configFactory) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === "undefined") return null;
    const existing = registry.get(canvasId);
    if (existing) existing.chart.destroy();
    const chart = new Chart(canvas.getContext("2d"), configFactory(palette()));
    registry.set(canvasId, { chart, configFactory });
    return chart;
  }

  const legends = new Map();

  function legend(containerId, entries) {
    const container = document.getElementById(containerId);
    if (!container) return;
    legends.set(containerId, entries);
    const colors = palette();
    container.innerHTML = entries
      .map((entry, index) => {
        const color = entry.color ? entry.color(colors) : colors.series[index % colors.series.length];
        return `<span><i style="background:${color}"></i>${entry.label}</span>`;
      })
      .join("");
  }

  document.addEventListener("cleartusk:themechange", () => {
    registry.forEach((entry, canvasId) => {
      const canvas = document.getElementById(canvasId);
      if (!canvas) return;
      entry.chart.destroy();
      entry.chart = new Chart(canvas.getContext("2d"), entry.configFactory(palette()));
    });
    legends.forEach((entries, containerId) => legend(containerId, entries));
  });

  window.ClearTuskCharts = { bars, lines, area, legend, palette, register };
})();
