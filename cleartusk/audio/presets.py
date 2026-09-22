"""Call-type profiles and denoiser presets.

Each elephant vocalisation occupies a different band, so the cleaner is
parameterised per call type. A *profile* fixes the protected band; a *preset*
describes how cautious one denoising pass should be. The pipeline evaluates
every preset in a profile and keeps the highest scoring candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

Band = tuple[float, float]

#: Band where machine noise (engines, rotors, generators) dominates.
MACHINE_BAND: Final[Band] = (150.0, 1000.0)
#: Narrow band used as a call-type independent sanity check.
CORE_RUMBLE_BAND: Final[Band] = (10.0, 150.0)


@dataclass(frozen=True, slots=True)
class Preset:
    """One denoising configuration.

    Attributes:
        hpss_margin: Harmonic/percussive separation margins ``(harmonic, percussive)``.
        threshold: SNR ratio below which the noise gate starts closing.
        softness: Width of the gate transition; larger is gentler.
        harmonic_blend: Fraction of raw harmonic content mixed back after masking.
        percussive_reject: Fraction of percussive content subtracted.
        target_reinject: Fraction of the original in-band call re-added.
    """

    name: str
    hpss_margin: Band
    threshold: float
    softness: float
    harmonic_blend: float
    percussive_reject: float
    target_reinject: float = 0.0


@dataclass(frozen=True, slots=True)
class CallProfile:
    """Everything the cleaner needs to know about one call type."""

    name: str
    display_name: str
    description: str
    target_band: Band
    freq_boost: float
    upper_taper_floor: float
    retention_floor: float
    presets: tuple[Preset, ...] = field(default_factory=tuple)

    @property
    def machine_band(self) -> Band:
        """Noise band scored for this profile, kept clear of the protected band."""
        low = max(MACHINE_BAND[0], self.target_band[1] + 25.0)
        high = MACHINE_BAND[1]
        if low >= high:
            low = max(MACHINE_BAND[0], high - 50.0)
        return float(low), float(high)


RUMBLE = CallProfile(
    name="rumble",
    display_name="Rumble",
    description="Low-frequency contact and social calls, 10-150 Hz. The most common call type.",
    target_band=(10.0, 150.0),
    freq_boost=1.4,
    upper_taper_floor=0.40,
    retention_floor=40.0,
    presets=(
        Preset("rumble_preserve", (1.1, 2.6), 1.08, 1.85, 0.36, 0.02, 0.26),
        Preset("rumble_conservative", (1.2, 3.0), 1.20, 1.50, 0.28, 0.03, 0.18),
        Preset("rumble_balanced", (1.5, 4.0), 1.35, 1.10, 0.15, 0.05, 0.10),
        Preset("rumble_aggressive", (1.8, 5.0), 1.50, 0.95, 0.10, 0.07, 0.06),
    ),
)

TRUMPET = CallProfile(
    name="trumpet",
    display_name="Trumpet",
    description="Bright, harmonically rich excitement calls reaching several hundred hertz.",
    target_band=(80.0, 450.0),
    freq_boost=1.2,
    upper_taper_floor=0.55,
    retention_floor=30.0,
    presets=(
        Preset("trumpet_preserve", (1.0, 2.3), 1.02, 1.90, 0.40, 0.02, 0.24),
        Preset("trumpet_conservative", (1.1, 2.5), 1.10, 1.70, 0.35, 0.02, 0.18),
        Preset("trumpet_balanced", (1.3, 3.2), 1.25, 1.25, 0.22, 0.04, 0.10),
    ),
)

ROAR = CallProfile(
    name="roar",
    display_name="Roar",
    description="Broadband distress and aggression calls spanning the mid frequencies.",
    target_band=(40.0, 300.0),
    freq_boost=1.25,
    upper_taper_floor=0.50,
    retention_floor=30.0,
    presets=(
        Preset("roar_preserve", (1.1, 2.7), 1.05, 1.85, 0.36, 0.02, 0.24),
        Preset("roar_conservative", (1.2, 3.0), 1.12, 1.55, 0.30, 0.03, 0.16),
        Preset("roar_balanced", (1.4, 3.8), 1.28, 1.20, 0.18, 0.05, 0.10),
    ),
)

DEFAULT = CallProfile(
    name="default",
    display_name="Mixed / unknown",
    description="Wide protected band used when the call type is unknown or mixed.",
    target_band=(15.0, 220.0),
    freq_boost=1.3,
    upper_taper_floor=0.45,
    retention_floor=30.0,
    presets=(
        Preset("default_preserve", (1.1, 2.6), 1.06, 1.85, 0.36, 0.02, 0.22),
        Preset("default_conservative", (1.2, 3.0), 1.18, 1.55, 0.28, 0.03, 0.16),
        Preset("default_balanced", (1.5, 4.0), 1.32, 1.20, 0.18, 0.05, 0.10),
    ),
)

CALL_PROFILES: Final[dict[str, CallProfile]] = {
    profile.name: profile for profile in (RUMBLE, TRUMPET, ROAR, DEFAULT)
}
#: Order used for dropdowns and reports.
CALL_TYPES: Final[tuple[str, ...]] = tuple(CALL_PROFILES)


def normalize_call_type(raw: str | None) -> str:
    """Map a free-text annotation such as ``"rumble-roar-rumble"`` to a profile name.

    Compound labels are resolved by specificity: the rarer, higher-frequency
    call wins, because preserving its band also preserves the rumble below it.
    """
    text = str(raw or "").strip().lower()
    for candidate in ("trumpet", "roar", "rumble"):
        if candidate in text:
            return candidate
    return "default"


def get_profile(call_type: str | None) -> CallProfile:
    """Return the profile for ``call_type``, falling back to the default profile."""
    key = str(call_type or "").strip().lower()
    return CALL_PROFILES.get(key, DEFAULT)
