document.addEventListener("DOMContentLoaded", () => {
  bindUploadForm();
});

function updateStatus(node, message, isError) {
  node.textContent = message;
  node.classList.toggle("text-danger", isError);
  node.classList.toggle("text-success", !isError);
}

function bindUploadForm() {
  const form = document.getElementById("upload-clean-form");
  if (!form) {
    return;
  }

  const statusNode = document.getElementById("upload-status");
  const submitButton = document.getElementById("upload-submit-btn");
  const resultsNode = document.getElementById("upload-results");
  const originalAudioPlayer = document.getElementById("original-audio-player");
  const cleanedAudioPlayer = document.getElementById("cleaned-audio-player");
  const originalSpectrogram = document.getElementById("original-spectrogram");
  const cleanedSpectrogram = document.getElementById("cleaned-spectrogram");
  const downloadLink = document.getElementById("download-cleaned-audio");
  const selectedPresetNode = document.getElementById("selected-preset");
  const metricsGrid = document.getElementById("metrics-grid");

  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const formData = new FormData(form);
    submitButton.disabled = true;
    submitButton.textContent = "Cleaning...";
    updateStatus(statusNode, "Uploading audio and generating spectrograms...", false);

    try {
      const response = await fetch("/process-upload", {
        method: "POST",
        body: formData,
      });
      const result = await response.json();

      if (!response.ok || !result.success) {
        throw new Error(result.error || "Upload processing failed.");
      }

      originalAudioPlayer.src = result.original_audio_url;
      cleanedAudioPlayer.src = result.cleaned_audio_url;
      originalSpectrogram.src = result.original_spectrogram_url;
      cleanedSpectrogram.src = result.cleaned_spectrogram_url;
      downloadLink.href = result.cleaned_audio_url;
      selectedPresetNode.textContent = `${result.selected_preset} (${result.call_type})`;

      metricsGrid.innerHTML = Object.entries(result.metrics)
        .map(
          ([label, value]) => `
            <div class="metric-card">
              <div class="metric-label">${label}</div>
              <div class="metric-value">${value}</div>
            </div>
          `
        )
        .join("");

      resultsNode.classList.remove("d-none");
      updateStatus(statusNode, result.message, false);
    } catch (error) {
      updateStatus(statusNode, error.message, true);
    } finally {
      submitButton.disabled = false;
      submitButton.textContent = "Upload And Clean";
    }
  });
}
