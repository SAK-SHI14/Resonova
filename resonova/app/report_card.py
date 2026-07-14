"""
Resonova — Emotion Report Card Generator
======================================
Generates a dark-themed visual summary (matplotlib / PIL) showing side-by-side
emotion classifications, acoustic feature comparison bars, and overlaid pitch contours.
"""

import io
import logging
from typing import Dict, Any, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def compute_preservation_score(
    prosody_original: Dict[str, Any],
    prosody_dubbed: Dict[str, Any]
) -> float:
    """
    Compute a single 0-100 preservation score from prosody features.

    Formula weights:
    - Emotion label match: 40 points
    - F0 mean proximity: 30 points
    - Energy proximity: 30 points
    """
    # 1. Emotion label match
    orig_emo = prosody_original.get("emotion_label", "").lower()
    dub_emo = prosody_dubbed.get("emotion_label", "").lower()
    emo_match = orig_emo == dub_emo and orig_emo != ""
    emo_points = 40.0 if emo_match else 0.0

    # 2. F0 mean proximity
    orig_f0 = prosody_original.get("f0_mean", 0.0)
    dub_f0 = prosody_dubbed.get("f0_mean", 0.0)
    if orig_f0 > 0 and dub_f0 > 0:
        f0_diff = abs(orig_f0 - dub_f0)
        f0_pct = max(0.0, 1.0 - (f0_diff / max(orig_f0, dub_f0, 1.0)))
        f0_points = 30.0 * f0_pct
    else:
        f0_points = 15.0  # Default neutral fallback

    # 3. Energy proximity
    orig_energy = prosody_original.get("energy_mean", 0.0)
    dub_energy = prosody_dubbed.get("energy_mean", 0.0)
    if orig_energy > 0 and dub_energy > 0:
        energy_diff = abs(orig_energy - dub_energy)
        energy_pct = max(0.0, 1.0 - (energy_diff / max(orig_energy, dub_energy, 0.01)))
        energy_points = 30.0 * energy_pct
    else:
        energy_points = 15.0  # Default neutral fallback

    score = emo_points + f0_points + energy_points
    return float(np.clip(score, 0.0, 100.0))


def generate_report_card(
    prosody_original: Dict[str, Any],
    prosody_dubbed: Dict[str, Any],
    speaker_similarity: float,
    processing_time_seconds: float,
    clip_duration_seconds: float
) -> Image.Image:
    """
    Generate the Resonova Emotion Analysis Report Card as a PIL Image.
    """
    # Initialize matplotlib figure
    fig = plt.figure(figsize=(11, 6), facecolor='#1a1a2e')
    
    # Base axes for drawing textual grid and card blocks
    ax_base = fig.add_axes([0, 0, 1, 1], facecolor='#1a1a2e')
    ax_base.set_facecolor('#1a1a2e')
    ax_base.axis('off')
    ax_base.set_xlim(0, 100)
    ax_base.set_ylim(0, 100)

    # Color palette
    PURPLE = '#7c3aed'
    GREEN = '#10b981'
    ORANGE = '#f59e0b'
    RED = '#ef4444'
    WHITE = '#f8fafc'
    GRAY = '#334155'
    LIGHT_GRAY = '#94a3b8'

    # ── 1. HEADER ─────────────────────────────────────────────────────────────
    ax_base.text(4, 94, "RESONOVA - EMOTION DUBBING", color=PURPLE, fontsize=12, fontweight='bold', alpha=0.8)
    ax_base.text(50, 92, "EMOTION ANALYSIS REPORT", color=WHITE, fontsize=16, fontweight='bold', ha='center')
    ax_base.text(50, 88, "English → Hindi Dubbing Verification", color=LIGHT_GRAY, fontsize=10, ha='center')

    # ── 2. TWO-COLUMN GRID BOUNDARIES ─────────────────────────────────────────
    # Left Box (Original)
    orig_rect = mpatches.FancyBboxPatch(
        (4, 45), 43, 38,
        boxstyle="round,pad=1.5",
        facecolor='#0f172a',
        edgecolor=GRAY,
        linewidth=1
    )
    ax_base.add_patch(orig_rect)

    # Right Box (Dubbed)
    dub_rect = mpatches.FancyBboxPatch(
        (53, 45), 43, 38,
        boxstyle="round,pad=1.5",
        facecolor='#0f172a',
        edgecolor=GRAY,
        linewidth=1
    )
    ax_base.add_patch(dub_rect)

    # Title labels inside boxes
    ax_base.text(6, 80, "ORIGINAL SPEAKER (English)", color=PURPLE, fontsize=11, fontweight='bold')
    ax_base.text(55, 80, "DUBBED OUTPUT (Hindi)", color=GREEN, fontsize=11, fontweight='bold')

    # Extract info
    orig_emo = prosody_original.get("emotion_label", "Neutral").capitalize()
    orig_conf = prosody_original.get("emotion_confidence", 0.85)
    dub_emo = prosody_dubbed.get("emotion_label", "Neutral").capitalize()
    dub_conf = prosody_dubbed.get("emotion_confidence", 0.85)

    ax_base.text(6, 75, f"Emotion: {orig_emo} ({orig_conf:.0%})", color=WHITE, fontsize=11)
    ax_base.text(55, 75, f"Emotion: {dub_emo} ({dub_conf:.0%})", color=WHITE, fontsize=11)

    # ── 3. FEATURE BAR COMPARISONS ────────────────────────────────────────────
    features_list = [
        # (key, display_name, max_val, unit)
        ("f0_mean", "Mean Pitch", 400.0, "Hz"),
        ("energy_mean", "Mean Energy", 0.15, "RMS"),
        ("speaking_rate", "Speaking Rate", 8.0, "syl/s")
    ]

    y_pos = 68
    for key, name, max_val, unit in features_list:
        val_orig = prosody_original.get(key, 0.0)
        val_dub = prosody_dubbed.get(key, 0.0)

        # Left Column text & bar
        ax_base.text(6, y_pos, f"{name}: {val_orig:.1f} {unit}", color=LIGHT_GRAY, fontsize=9.5)
        # Background bar
        bg_bar_orig = mpatches.Rectangle((6, y_pos - 3.5), 35, 1.8, facecolor=GRAY, alpha=0.3)
        ax_base.add_patch(bg_bar_orig)
        # Foreground filled bar
        len_orig = 35.0 * (min(val_orig, max_val) / max_val) if max_val > 0 else 0.0
        fill_bar_orig = mpatches.Rectangle((6, y_pos - 3.5), len_orig, 1.8, facecolor=PURPLE)
        ax_base.add_patch(fill_bar_orig)

        # Right Column text & bar
        ax_base.text(55, y_pos, f"{name}: {val_dub:.1f} {unit}", color=LIGHT_GRAY, fontsize=9.5)
        # Background bar
        bg_bar_dub = mpatches.Rectangle((55, y_pos - 3.5), 35, 1.8, facecolor=GRAY, alpha=0.3)
        ax_base.add_patch(bg_bar_dub)
        # Foreground filled bar
        len_dub = 35.0 * (min(val_dub, max_val) / max_val) if max_val > 0 else 0.0
        fill_bar_dub = mpatches.Rectangle((55, y_pos - 3.5), len_dub, 1.8, facecolor=GREEN)
        ax_base.add_patch(fill_bar_dub)

        y_pos -= 8.5

    # ── 4. EMOTION PRESERVATION STATUS ────────────────────────────────────────
    is_match = orig_emo.lower() == dub_emo.lower()
    status_text = "EMOTION PRESERVED" if is_match else "EMOTION SHIFTED"
    status_color = GREEN if is_match else ORANGE
    ax_base.text(50, 47, status_text, color=status_color, fontsize=11, fontweight='bold', ha='center')

    # ── 5. BOTTOM METRIC BOXES ────────────────────────────────────────────────
    # Preservation Score Box
    score_box = mpatches.FancyBboxPatch(
        (4, 8), 17, 27,
        boxstyle="round,pad=1.0",
        facecolor='#0f172a',
        edgecolor=GRAY,
        linewidth=1
    )
    ax_base.add_patch(score_box)
    
    score = compute_preservation_score(prosody_original, prosody_dubbed)
    score_color = GREEN if score >= 75.0 else (ORANGE if score >= 50.0 else RED)
    ax_base.text(12.5, 30, "Preservation Score", color=LIGHT_GRAY, fontsize=8.5, ha='center')
    ax_base.text(12.5, 18, f"{score:.1f}%", color=score_color, fontsize=20, fontweight='bold', ha='center')
    score_bar_bg = mpatches.Rectangle((7, 12), 11, 1.5, facecolor=GRAY, alpha=0.3)
    ax_base.add_patch(score_bar_bg)
    score_bar_fill = mpatches.Rectangle((7, 12), 11 * (score / 100.0), 1.5, facecolor=score_color)
    ax_base.add_patch(score_bar_fill)

    # Speaker Similarity Box
    sim_box = mpatches.FancyBboxPatch(
        (24, 8), 17, 27,
        boxstyle="round,pad=1.0",
        facecolor='#0f172a',
        edgecolor=GRAY,
        linewidth=1
    )
    ax_base.add_patch(sim_box)
    ax_base.text(32.5, 30, "Speaker Similarity", color=LIGHT_GRAY, fontsize=8.5, ha='center')
    ax_base.text(32.5, 18, f"{speaker_similarity:.1%}", color=GREEN, fontsize=20, fontweight='bold', ha='center')
    sim_bar_bg = mpatches.Rectangle((27, 12), 11, 1.5, facecolor=GRAY, alpha=0.3)
    ax_base.add_patch(sim_bar_bg)
    sim_bar_fill = mpatches.Rectangle((27, 12), 11 * speaker_similarity, 1.5, facecolor=GREEN)
    ax_base.add_patch(sim_bar_fill)

    # Processing Time Box
    time_box = mpatches.FancyBboxPatch(
        (44, 8), 17, 27,
        boxstyle="round,pad=1.0",
        facecolor='#0f172a',
        edgecolor=GRAY,
        linewidth=1
    )
    ax_base.add_patch(time_box)
    ax_base.text(52.5, 30, "Processing Performance", color=LIGHT_GRAY, fontsize=8.5, ha='center')
    ax_base.text(52.5, 20, f"{processing_time_seconds:.0f}s", color=WHITE, fontsize=18, fontweight='bold', ha='center')
    ax_base.text(52.5, 13, f"for {clip_duration_seconds:.1f}s clip", color=LIGHT_GRAY, fontsize=8, ha='center')

    # ── 6. PITCH CONTOUR PLOT (Bottom Right Subplot) ──────────────────────────
    orig_f0_contour = prosody_original.get("pitch_contour")
    dub_f0_contour = prosody_dubbed.get("pitch_contour")

    if orig_f0_contour and dub_f0_contour:
        # Create an inset axes for the pitch contour plot
        # x_start, y_start, width, height relative to figure [0, 1]
        ax_plot = fig.add_axes([0.66, 0.08, 0.28, 0.28])
        ax_plot.set_facecolor('#0f172a')
        
        # Resample contours to a standard size for plotting
        target_len = 100
        # Simple resampler
        def resample(arr, size):
            old_indices = np.linspace(0, 1, len(arr))
            new_indices = np.linspace(0, 1, size)
            return np.interp(new_indices, old_indices, arr)

        orig_y = resample(orig_f0_contour, target_len)
        dub_y = resample(dub_f0_contour, target_len)

        # Plot contours
        ax_plot.plot(orig_y, color=PURPLE, label='Orig', linewidth=1.5)
        ax_plot.plot(dub_y, color=GREEN, label='Dub', linewidth=1.5)
        
        ax_plot.set_title("Pitch Contour Comparison", color=WHITE, fontsize=8.5, pad=4)
        ax_plot.spines['bottom'].set_color(GRAY)
        ax_plot.spines['left'].set_color(GRAY)
        ax_plot.spines['top'].set_visible(False)
        ax_plot.spines['right'].set_visible(False)
        ax_plot.tick_params(colors=LIGHT_GRAY, labelsize=7)
        ax_plot.grid(color=GRAY, linestyle='--', linewidth=0.5, alpha=0.3)
        ax_plot.legend(facecolor='#0f172a', edgecolor='none', labelcolor=WHITE, fontsize=7, loc='upper right')
    else:
        # Graceful placeholder inside base coordinates if contours are not available
        contour_placeholder = mpatches.FancyBboxPatch(
            (64, 8), 32, 27,
            boxstyle="round,pad=1.0",
            facecolor='#0f172a',
            edgecolor=GRAY,
            linewidth=1
        )
        ax_base.add_patch(contour_placeholder)
        ax_base.text(80, 22, "Pitch Contour", color=WHITE, fontsize=9.5, fontweight='bold', ha='center')
        ax_base.text(80, 16, "Comparison Plot", color=WHITE, fontsize=9.5, fontweight='bold', ha='center')
        ax_base.text(80, 11, "(No Contour Data Available)", color=LIGHT_GRAY, fontsize=7.5, ha='center')

    # Convert plot figure to PIL Image
    buf = io.BytesIO()
    plt.savefig(
        buf,
        format='PNG',
        bbox_inches='tight',
        facecolor=fig.get_facecolor(),
        dpi=120
    )
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img
