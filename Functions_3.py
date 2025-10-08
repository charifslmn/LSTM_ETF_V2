#------------------------------------------------------------------------------------------------------------------------
#                                             ABOUT 

#       - funtions used in the (3) CLEAN Model Performance Assesment HOD 20_01 - 21_12.ipynb 
#       - used to analye the performance of different models and seeds in clean manner 

#------------------------------------------------------------------------------------------------------------------------

    

import math
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Data helpers
# -----------------------------
def _flatten_list_of_lists(lol):
    """Flatten a list of lists; ignore None and non-iterables safely."""
    out = []
    if not lol:
        return out
    for sub in lol:
        if sub is None:
            continue
        # assume sub is iterable of numbers
        try:
            out.extend(sub)
        except TypeError:
            # if sub isn't iterable, treat as single value
            out.append(sub)
    return out

def _nan_to_num_list(vals):
    """Replace NaN/inf with 0 for stable sums; drop None."""
    clean = []
    for v in vals:
        if v is None:
            continue
        try:
            vf = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(vf):
            vf = 0.0
        clean.append(vf)
    return clean

def compute_six_metrics(entry):
    """
    entry: dict with keys:
      - "REG_UNIQUE_ALL": list[list[number]]
      - "THR_UNIQUE_ALL": list[list[number]]
    Returns (reg_sum, reg_cnt, reg_ret, thr_sum, thr_cnt, thr_ret)
    where *_ret = sum(unique)/count(unique) (0 if count==0).
    """
    reg_flat = _nan_to_num_list(_flatten_list_of_lists(entry.get("REG_UNIQUE_ALL")))
    thr_flat = _nan_to_num_list(_flatten_list_of_lists(entry.get("THR_UNIQUE_ALL")))

    reg_unique = set(reg_flat)
    thr_unique = set(thr_flat)

    reg_sum = sum(reg_unique)
    reg_cnt = len(reg_unique)
    reg_ret = (reg_sum / reg_cnt) if reg_cnt > 0 else 0.0

    thr_sum = sum(thr_unique)
    thr_cnt = len(thr_unique)
    thr_ret = (thr_sum / thr_cnt) if thr_cnt > 0 else 0.0

    return reg_sum, reg_cnt, reg_ret, thr_sum, thr_cnt, thr_ret


# -----------------------------
# Plotting
# -----------------------------
def plot_model_six_bars(
    models_dict,
    title="Unique-Value Metrics per Model",
    annotate=True,
    annotate_position="above",  # "above" or "below"
    xtick_rotation=45,
    save_path=None
):
    """
    models_dict: dict like
        {
          "ModelA": {"REG_UNIQUE_ALL":[[...], ...], "THR_UNIQUE_ALL":[[...], ...]},
          "ModelB": {...},
          ...
        }
    annotate_position:
        - "above": values above each bar
        - "below": all values aligned along a single baseline under the bars
    """
    # 1) Compute matrix (n_models x 6)
    model_names = list(models_dict.keys())
    metrics = [compute_six_metrics(models_dict[name]) for name in model_names]
    data = np.array(metrics, dtype=float) if metrics else np.zeros((0, 6), dtype=float)

    labels = ["REG_SUM", "REG_NUM_PREDS", "REG_RETURN",
              "THR_SUM", "THR_NUM_PREDS", "THR_RETURN"]

    n_models = len(model_names)
    n_bars = 6
    x = np.arange(n_models, dtype=float)

    # 2) Layout
    width = min(0.12, 0.7 / n_bars)  # keep groups compact
    fig_w = max(10, n_models * 1.9)
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    # 3) Bars
    series = []
    for i in range(n_bars):
        series.append(ax.bar(x + i * width, data[:, i], width, label=labels[i]))

    # 4) Axes, ticks, legend
    ax.set_xticks(x + width * (n_bars - 1) / 2)
    ax.set_xticklabels(model_names, rotation=xtick_rotation, ha="right" if xtick_rotation else "center")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend(ncols=3, frameon=False)
    ax.margins(x=0.02)

    # 5) Annotations
    if annotate:
        if annotate_position.lower() == "below":
            # One common baseline under all bars
            ymin, ymax = ax.get_ylim()
            yr = ymax - ymin if ymax > ymin else 1.0
            # Make extra space at bottom so text isn't clipped
            ax.set_ylim(ymin - 0.2 * yr, ymax)
            y0, y1 = ax.get_ylim()
            y_text = y0 + 0.04 * (y1 - y0)  # common baseline

            for i in range(n_bars):
                for rect in series[i]:
                    h = rect.get_height()
                    # We annotate the numeric value but position it at y_text
                    x_center = rect.get_x() + rect.get_width() / 2.0
                    ax.annotate(f"{h:.3g}",
                                xy=(x_center, y_text),
                                xytext=(0, 15),
                                rotation = 85 , 
                                rotation_mode = "anchor" , 
                                textcoords="offset points",
                                ha='center', va='top', fontsize=10)


    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")

    plt.show()


# # -----------------------------
# # Example usage
# # -----------------------------

# plot_model_six_bars(
#     data,
#     title="Unique-Value Metrics per Model",
#     annotate=True,
#     annotate_position="below",   
#     xtick_rotation=45,
#     save_path=None               
# )
