import numpy as np
import pytest

from cleartusk.audio.denoise import clean_signal, isolate_call
from cleartusk.audio.features import FEATURE_DIM, extract_features, infer_call_type
from cleartusk.audio.io import probe, stft_params
from cleartusk.audio.metrics import band_energy, compute_metrics, score_candidate
from cleartusk.audio.presets import CALL_PROFILES, get_profile, normalize_call_type
from cleartusk.audio.spectrogram import compute_spectrogram, render_comparison, render_spectrogram
from tests.conftest import SAMPLE_RATE, synth_call, synth_noise


def test_normalize_call_type_resolves_compound_labels():
    assert normalize_call_type("rumble-roar-rumble") == "roar"
    assert normalize_call_type("trumpet-rumble") == "trumpet"
    assert normalize_call_type("bark-rumble") == "rumble"
    assert normalize_call_type("") == "default"
    assert normalize_call_type(None) == "default"


def test_machine_band_never_overlaps_the_protected_band():
    for profile in CALL_PROFILES.values():
        assert profile.machine_band[0] > profile.target_band[1]
        assert profile.machine_band[0] < profile.machine_band[1]


def test_cleaning_removes_machine_energy_and_keeps_the_call(settings):
    audio = synth_call()
    profile = get_profile("rumble")
    result = clean_signal(audio, profile)

    assert result.metrics.machine_suppression_pct > 60
    assert result.metrics.target_retention_pct > 15
    assert result.metrics.spectral_fidelity > 0.5
    assert result.metrics.target_machine_gain_db > 0
    assert result.candidates_evaluated == len(profile.presets)
    assert np.max(np.abs(result.audio)) <= 1.0


def test_retention_is_capped_at_the_original_energy(settings):
    audio = synth_call()
    profile = get_profile("rumble")
    for preset in profile.presets:
        cleaned = isolate_call(audio, profile, preset)
        original = band_energy(audio, SAMPLE_RATE, profile.target_band)
        assert band_energy(cleaned, SAMPLE_RATE, profile.target_band) <= original * 1.001


def test_metrics_stay_inside_their_definitions(settings):
    audio = synth_call()
    profile = get_profile("rumble")
    metrics = compute_metrics(audio, clean_signal(audio, profile).audio, profile, SAMPLE_RATE)

    assert 0 <= metrics.target_retention_pct <= 100
    assert metrics.machine_suppression_pct <= 100
    assert -1 <= metrics.spectral_fidelity <= 1
    assert 0 <= metrics.spectral_sparsity_pct <= 100
    assert metrics.target_band_low_hz == profile.target_band[0]


def test_score_prefers_the_candidate_that_keeps_more_call(settings):
    audio = synth_call()
    profile = get_profile("rumble")
    cleaned = clean_signal(audio, profile).audio
    good = compute_metrics(audio, cleaned, profile, SAMPLE_RATE)
    hollow = compute_metrics(audio, cleaned * 0.01, profile, SAMPLE_RATE)
    assert score_candidate(good, profile) > score_candidate(hollow, profile)


def test_empty_signal_is_rejected(settings):
    with pytest.raises(ValueError):
        clean_signal(np.zeros(0, dtype=np.float32), get_profile("rumble"))


def test_feature_vector_is_fixed_length_and_finite():
    features = extract_features(synth_call(), SAMPLE_RATE)
    assert features.shape == (FEATURE_DIM,)
    assert np.isfinite(features).all()


def test_short_windows_are_padded_not_rejected():
    assert extract_features(np.zeros(120, dtype=np.float32), SAMPLE_RATE).shape == (FEATURE_DIM,)


def test_call_type_inference_follows_the_spectral_mass():
    t = np.linspace(0, 2, 2 * SAMPLE_RATE, endpoint=False)
    assert infer_call_type(np.sin(2 * np.pi * 30 * t).astype(np.float32), SAMPLE_RATE)[0] == "rumble"
    assert infer_call_type(np.sin(2 * np.pi * 520 * t).astype(np.float32), SAMPLE_RATE)[0] == "trumpet"


def test_stft_parameters_shrink_for_short_signals():
    n_fft, hop, win = stft_params(300)
    assert n_fft <= 300 or n_fft == 256
    assert hop >= 32 and win <= n_fft


def test_spectrograms_render_to_disk(settings, tmp_path):
    audio = synth_call()
    cleaned = clean_signal(audio, get_profile("rumble")).audio

    single = render_spectrogram(audio, SAMPLE_RATE, tmp_path / "one.png", events=[(0.5, 1.5)])
    both = render_comparison(audio, cleaned, SAMPLE_RATE, tmp_path / "two.png")

    assert single.stat().st_size > 1000
    assert both.stat().st_size > 1000
    assert compute_spectrogram(audio, SAMPLE_RATE).duration_seconds == pytest.approx(3.0, abs=0.05)


def test_probe_reports_duration(tmp_path):
    from tests.conftest import write_wav

    path = tmp_path / "probe.wav"
    write_wav(path, synth_noise(seconds=2.0))
    info = probe(path)
    assert info.duration_seconds == pytest.approx(2.0, abs=0.05)
    assert info.sample_rate == SAMPLE_RATE
