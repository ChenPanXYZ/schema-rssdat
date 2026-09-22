import math, argparse, os, re, torch, random
from transformers import AutoModelForCausalLM, AutoTokenizer
from model_downloader import download_model
from dotenv import load_dotenv
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

load_dotenv()

ROOT_SEED = 0
TICK_LABELSIZE = 12
SINGLE_FIG_PAD = 2.0
COMBINED_FIG_PAD = 0.4
SUBPLOT_TITLE_PAD_MULT = 2.2
COMBINED_SUPTITLE_Y = 0.99
COMBINED_TIGHT_LAYOUT_RECT_TOP = 0.97


_MPL_FONT_KEYS_TO_SCALE = (
    "font.size",
    "axes.titlesize",
    # "axes.labelsize",
    # "xtick.labelsize",
    # "ytick.labelsize",
    # "legend.fontsize",
    "figure.titlesize",
    "axes.titlepad",
)
_ORIG_MPL_RCPARAMS = {k: plt.rcParams.get(k) for k in _MPL_FONT_KEYS_TO_SCALE}


def set_plot_text_scale(scale: float = 2.0) -> None:
    """
    Scale all plot text sizes (titles, axis labels, tick labels, legends, etc.).
    """
    scale = float(scale)
    if scale <= 0:
        raise ValueError(f"scale must be > 0; got {scale}")

    for k in _MPL_FONT_KEYS_TO_SCALE:
        v = _ORIG_MPL_RCPARAMS.get(k)
        if isinstance(v, (int, float)):
            plt.rcParams[k] = v * scale

    # Slightly reduce tick label font size vs other text.
    tick_scale = 0.85
    for k in ("xtick.labelsize", "ytick.labelsize"):
        v = _ORIG_MPL_RCPARAMS.get(k)
        if isinstance(v, (int, float)):
            plt.rcParams[k] = v * scale * tick_scale

    # Seaborn applies its own defaults; keep heatmap text consistent.
    sns.set_context("notebook", font_scale=scale)


def _apply_tick_labelsize(ax, size: int = TICK_LABELSIZE) -> None:
    if ax is None:
        return
    ax.tick_params(axis="both", which="both", labelsize=int(size))
    # Force tick labels to stay horizontal (no auto-rotation).
    ax.tick_params(axis="x", which="both", labelrotation=0)
    ax.tick_params(axis="y", which="both", labelrotation=0)
    for lab in ax.get_xticklabels():
        lab.set_rotation(0)
        lab.set_horizontalalignment("center")
    for lab in ax.get_yticklabels():
        lab.set_rotation(0)
        lab.set_verticalalignment("center")
    # If this axes has a seaborn/matplotlib colorbar, scale its ticks too.
    try:
        if getattr(ax, "collections", None):
            cb = ax.collections[0].colorbar
            if cb is not None and getattr(cb, "ax", None) is not None:
                cb.ax.tick_params(axis="both", which="both", labelsize=int(size))
                cb.ax.tick_params(axis="x", which="both", labelrotation=0)
                cb.ax.tick_params(axis="y", which="both", labelrotation=0)
    except Exception:
        # Best-effort; don't fail plotting if colorbar internals differ.
        pass


def _subplot_title_pad_points() -> float:
    base = plt.rcParams.get("axes.titlepad", 6.0)
    try:
        base_f = float(base)
    except Exception:
        base_f = 6.0
    return base_f * float(SUBPLOT_TITLE_PAD_MULT)


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def sample_run_seeds(root_seed: int, n: int) -> list[int]:
    if n <= 0:
        raise ValueError(f"n must be > 0; got {n}")
    rng = random.Random(int(root_seed))
    # Deterministically sample seeds from ROOT_SEED.
    return [rng.randrange(0, 2**31 - 1) for _ in range(int(n))]

prompt = """Answer with only one letter: A.Laundry and B.Dishwashing, choose one.

The procedure is actually quite simple. First you arrange items into different groups. Of course,one pile may be sufficient depending on how much there is to do. If you have to go somewhereelse due to lack of facilities that is the next step, otherwise, you are pretty well set. It is important not to overdo things. That is, it is better to do too few things at once than too many. In the short run this may not seem important, but complications can easily arise. A mistake can be expensive as well. At first, the whole procedure will seem complicated. Soon, however, it will become just another facet of life. It is difficult to foresee any end to the necessity for this task in the immediate future, but then, one never can tell. After the procedure is completed one arranges the materials into different groups again. Then they can be put into their appropriate places. Eventually they will be used once more and the whole cycle will then have to be repeated. However, this is part of life."""

washing_machine_prompt = """Answer with only one letter: A.Laundry and B.Dishwashing, choose one.

Washing Machine. The procedure is actually quite simple. First you arrange items into different groups. Of course,one pile may be sufficient depending on how much there is to do. If you have to go somewhereelse due to lack of facilities that is the next step, otherwise, you are pretty well set. It is important not to overdo things. That is, it is better to do too few things at once than too many. In the short run this may not seem important, but complications can easily arise. A mistake can be expensive as well. At first, the whole procedure will seem complicated. Soon, however, it will become just another facet of life. It is difficult to foresee any end to the necessity for this task in the immediate future, but then, one never can tell. After the procedure is completed one arranges the materials into different groups again. Then they can be put into their appropriate places. Eventually they will be used once more and the whole cycle will then have to be repeated. However, this is part of life."""

kitchen_prompt = """Answer with only one letter: A.Laundry and B.Dishwashing, choose one.

Kitchen. The procedure is actually quite simple. First you arrange items into different groups. Of course,one pile may be sufficient depending on how much there is to do. If you have to go somewhereelse due to lack of facilities that is the next step, otherwise, you are pretty well set. It is important not to overdo things. That is, it is better to do too few things at once than too many. In the short run this may not seem important, but complications can easily arise. A mistake can be expensive as well. At first, the whole procedure will seem complicated. Soon, however, it will become just another facet of life. It is difficult to foresee any end to the necessity for this task in the immediate future, but then, one never can tell. After the procedure is completed one arranges the materials into different groups again. Then they can be put into their appropriate places. Eventually they will be used once more and the whole cycle will then have to be repeated. However, this is part of life."""

def _output_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "outputs")

def _clean_token_for_display(t: str) -> str:
    # Remove common tokenizer whitespace markers
    if t == "<|begin_of_text|>":
        return "<BOS>"
    return t.replace("▁", "").lstrip("Ġ")

def _normalize_token_for_matching(t: str) -> str:
    # Normalize to alphanumerics only for robust matching of "A"/"B" across tokenizers.
    t = _clean_token_for_display(t).strip()
    t = re.sub(r"[^A-Za-z0-9]", "", t)
    return t

def _find_first_option_letter_token_index(tokens_list, letter: str) -> int:
    target = letter.strip()
    for i, tok in enumerate(tokens_list):
        if _normalize_token_for_matching(tok) == target:
            return i
    raise ValueError(f"Could not find option letter token '{letter}' in tokenized prompt.")

def _resolve_vmin_vmax(
    tensors: list[torch.Tensor],
    *,
    vmin: float | None,
    vmax: float | None,
    clip_pct: float | None,
) -> tuple[float | None, float | None]:
    """
    Decide the heatmap color scale.

    - If vmin/vmax are provided, use them as-is.
    - Else if clip_pct is set, compute (clip_pct, 1-clip_pct) quantiles across all provided tensors.
    - Else return (None, None) to let seaborn auto-scale each plot.
    """
    if vmin is not None or vmax is not None:
        return vmin, vmax

    if clip_pct is None:
        return None, None

    if not (0.0 <= clip_pct < 0.5):
        raise ValueError(f"clip_pct must be in [0, 0.5); got {clip_pct}")

    flat = torch.cat([t.detach().float().reshape(-1).cpu() for t in tensors], dim=0)
    if flat.numel() == 0:
        return None, None

    lo = float(torch.quantile(flat, clip_pct).item())
    hi = float(torch.quantile(flat, 1.0 - clip_pct).item())
    return lo, hi


def _layers_heads_attention_last_token_to_target(
    attention_map: list[torch.Tensor],
    token_index: int,
) -> torch.Tensor:
    """
    Return a (layers x heads) tensor of attention weight from the last token
    to a target token at `token_index`.

    attention_map: list of [batch, heads, seq_len, seq_len]
    """
    num_layers = len(attention_map)
    if num_layers == 0:
        raise ValueError("attention_map is empty.")
    per_layer = []
    for layer_attn in attention_map:
        layer_attn = layer_attn[0]  # [heads, seq_len, seq_len]
        per_layer.append(layer_attn[:, -1, token_index].detach().cpu())
    return torch.stack(per_layer, dim=0)  # [layers, heads]

def _entropy_bits_of_scores(x: torch.Tensor) -> float:
    """
    Entropy (in bits) of all scores in x, treated as a discrete distribution
    after normalizing x.flatten() to sum to 1.

    Note: The values in `x` are not inherently a probability distribution across
    layers/heads; this is a descriptive statistic over the heatmap values.
    """
    flat = x.detach().float().reshape(-1).cpu()
    if flat.numel() == 0:
        return float("nan")
    flat = torch.clamp(flat, min=0.0)
    s = float(flat.sum().item())
    if s <= 0.0:
        return 0.0
    p = flat / s
    # Mask zeros to avoid log(0)
    p = p[p > 0]
    h_nats = -torch.sum(p * torch.log(p)).item()
    h_bits = float(h_nats / math.log(2.0))
    return h_bits


def _topk_weight_ratio(x_layers_heads: torch.Tensor, k: int) -> float:
    """
    Score = sum(Top k Attention Weights) / sum(All Weights)
    computed over all entries in the (layers x heads) matrix.
    """
    if k <= 0:
        raise ValueError(f"k must be > 0; got {k}")
    if x_layers_heads.numel() == 0:
        return float("nan")
    flat = x_layers_heads.detach().float().reshape(-1).cpu()
    flat = torch.clamp(flat, min=0.0)
    denom = float(flat.sum().item())
    if denom <= 0.0:
        return 0.0
    k_eff = min(int(k), int(flat.numel()))
    topk_vals, _ = torch.topk(flat, k_eff)
    numer = float(topk_vals.sum().item())
    return numer / denom


def _plot_label_view_heatmap(
    data_layers_heads: torch.Tensor,
    *,
    ax,
    title: str,
    vmin: float,
    vmax: float,
    topk_weights: int = 10,
    entropy_bits: float | None = None,
    entropy_bits_std: float | None = None,
    score: float | None = None,
    score_std: float | None = None,
):
    num_layers, num_heads = data_layers_heads.shape
    white_to_red = LinearSegmentedColormap.from_list("white_to_red", ["#ffffff", "#d10000"])
    sns.heatmap(
        data_layers_heads.float().numpy(),
        cmap=white_to_red,
        vmin=vmin,
        vmax=vmax,
        xticklabels=list(range(num_heads)),
        yticklabels=list(range(num_layers)),
        ax=ax,
    )
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_title(title, pad=_subplot_title_pad_points())
    _apply_tick_labelsize(ax, TICK_LABELSIZE)
    h_bits = float(entropy_bits) if entropy_bits is not None else _entropy_bits_of_scores(data_layers_heads)
    score_val = float(score) if score is not None else _topk_weight_ratio(data_layers_heads, topk_weights)

    if entropy_bits_std is not None:
        h_text = f"H={h_bits:.3f}±{float(entropy_bits_std):.3f} bits"
    else:
        h_text = f"H={h_bits:.3f} bits"

    if score_std is not None:
        s_text = f"Score(top{topk_weights})={score_val:.4f}±{float(score_std):.4f}"
    else:
        s_text = f"Score(top{topk_weights})={score_val:.4f}"
    ax.text(
        0.99,
        0.99,
        f"{h_text}\n{s_text}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=18,
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
    )


def plot_label_views_combined(
    *,
    label: str,
    datas: list[torch.Tensor],
    subplot_titles: list[str],
    out_path: str,
    heatmap_vmin: float | None = None,
    heatmap_vmax: float | None = None,
    heatmap_clip_pct: float | None = None,
    topk_weights: int = 10,
    subplot_metrics: list[dict] | None = None,
):
    """
    Create a single figure for one label (A or B) with 3 subplots (one per prompt).
    """
    if len(datas) == 0:
        raise ValueError("datas is empty.")
    if len(datas) != len(subplot_titles):
        raise ValueError("datas and subplot_titles must have the same length.")
    if subplot_metrics is not None and len(subplot_metrics) != len(datas):
        raise ValueError("subplot_metrics must be the same length as datas when provided.")

    # Shared scale across the subplots unless the user overrides.
    vmin, vmax = _resolve_vmin_vmax(
        datas,
        vmin=heatmap_vmin,
        vmax=heatmap_vmax,
        clip_pct=heatmap_clip_pct,
    )
    if vmin is None and vmax is None:
        vmin = min(float(t.min().item()) for t in datas)
        vmax = max(float(t.max().item()) for t in datas)
        if vmin == vmax:
            eps = 1e-12
            vmin -= eps
            vmax += eps

    n = len(datas)
    num_cols = n
    num_rows = 1
    num_layers, num_heads = datas[0].shape
    panel_w = max(6.0, 0.35 * num_heads)
    panel_h = max(4.0, 0.25 * num_layers)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(panel_w * num_cols, panel_h), squeeze=False)

    for i, (data, st) in enumerate(zip(datas, subplot_titles)):
        ax = axes[0][i]
        metrics = subplot_metrics[i] if subplot_metrics is not None else None
        _plot_label_view_heatmap(
            data,
            ax=ax,
            title=st,
            vmin=vmin,
            vmax=vmax,
            topk_weights=topk_weights,
            entropy_bits=(metrics.get("entropy_mean") if metrics else None),
            entropy_bits_std=(metrics.get("entropy_std") if metrics else None),
            score=(metrics.get("score_mean") if metrics else None),
            score_std=(metrics.get("score_std") if metrics else None),
        )

    fig.suptitle(f"Attention(last token → '{label}') across layers/heads", y=COMBINED_SUPTITLE_Y)
    # Keep the suptitle close to the subplots (minimal reserved top margin).
    fig.tight_layout(pad=COMBINED_FIG_PAD, rect=(0, 0, 1, COMBINED_TIGHT_LAYOUT_RECT_TOP))
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_label_view_single(
    *,
    label: str,
    data: torch.Tensor,
    out_path: str,
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
    clip_pct: float | None = None,
    topk_weights: int = 10,
    metrics: dict | None = None,
):
    """
    Save a single heatmap figure for one label (A/B) and one prompt variant.
    """
    vmin, vmax = _resolve_vmin_vmax([data], vmin=vmin, vmax=vmax, clip_pct=clip_pct)
    if vmin is None and vmax is None:
        vmin = float(data.min().item())
        vmax = float(data.max().item())
        if vmin == vmax:
            eps = 1e-12
            vmin -= eps
            vmax += eps

    num_layers, num_heads = data.shape
    fig_w = max(6.0, 0.35 * num_heads)
    fig_h = max(4.0, 0.25 * num_layers)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    _plot_label_view_heatmap(
        data,
        ax=ax,
        title=title,
        vmin=vmin,
        vmax=vmax,
        topk_weights=topk_weights,
        entropy_bits=(metrics.get("entropy_mean") if metrics else None),
        entropy_bits_std=(metrics.get("entropy_std") if metrics else None),
        score=(metrics.get("score_mean") if metrics else None),
        score_std=(metrics.get("score_std") if metrics else None),
    )
    fig.tight_layout(pad=SINGLE_FIG_PAD)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_label_view_last_token_to_target(
    attention_map,
    token_index: int,
    tokens_list,
    out_path: str,
    token_label: str,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    clip_pct: float | None = None,
):
    """
    Build a (layers x heads) heatmap of attention weight from the last token
    to a target token at `token_index`.
    """
    data = _layers_heads_attention_last_token_to_target(attention_map, token_index)
    num_layers, num_heads = data.shape

    vmin, vmax = _resolve_vmin_vmax([data], vmin=vmin, vmax=vmax, clip_pct=clip_pct)
    # For the per-label view, ensure min is white and max is red by spanning
    # the full colormap range for each label's plot (unless the user overrides).
    if vmin is None and vmax is None:
        vmin = float(data.min().item())
        vmax = float(data.max().item())
        if vmin == vmax:
            # Avoid a degenerate colormap scale for constant tensors.
            eps = 1e-12
            vmin -= eps
            vmax += eps

    fig_w = max(6.0, 0.35 * num_heads)
    fig_h = max(4.0, 0.25 * num_layers)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    _plot_label_view_heatmap(
        data,
        ax=ax,
        title=f"Attention(last token → '{token_label}') across layers/heads",
        vmin=vmin,
        vmax=vmax,
    )
    fig.tight_layout(pad=SINGLE_FIG_PAD)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def plot_attn_map_triangle(layers, layer_offset: int, tokens_list, out_path: str, num_cols: int = 4):
    num_layers = len(layers)
    num_cols = max(1, min(num_cols, num_layers))
    num_rows = math.ceil(num_layers / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx in tqdm(range(num_layers)):
        ax = axes_flat[idx]
        avg_attention_scores = layers[idx][0].mean(dim=0)    # [ seq_len, seq_len]
        mask = torch.triu(torch.ones_like(avg_attention_scores, dtype=torch.bool), diagonal=1)
        sns.heatmap(
            avg_attention_scores.float().cpu().numpy(),
            mask=mask.cpu().numpy(),
            cmap='RdBu_r',
            square=True,
            xticklabels=tokens_list,
            yticklabels=tokens_list,
            ax=ax,
        )
        ax.set_title(f'layer {layer_offset + idx}', pad=_subplot_title_pad_points())
        _apply_tick_labelsize(ax, TICK_LABELSIZE)

    for ax in axes_flat[num_layers:]:
        ax.axis("off")

    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_attn_map_last_row_individual_head(layers, layer_offset: int, tokens_list, out_path: str, num_cols: int = 4):
    num_layers = len(layers)
    num_cols = max(1, min(num_cols, num_layers))
    num_rows = math.ceil(num_layers / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx in tqdm(range(num_layers)):
        ax = axes_flat[idx]
        # layers[idx]: [batch, heads, seq_len, seq_len]
        layer_attn = layers[idx][0]  # [heads, seq_len, seq_len]
        last_row = layer_attn[:, -1, :]       # [heads, seq_len]

        sns.heatmap(
            last_row.float().cpu().numpy(),
            cmap='RdBu_r',
            square=False,
            xticklabels=tokens_list,
            yticklabels=list(range(last_row.shape[0])),
            ax=ax,
        )
        ax.set_title(f'layer {layer_offset + idx} (last row, per head)', pad=_subplot_title_pad_points())
        ax.set_ylabel("head")
        _apply_tick_labelsize(ax, TICK_LABELSIZE)

    for ax in axes_flat[num_layers:]:
        ax.axis("off")

    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def visualize_attention_map(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, question: str):
    inputs = tokenizer(question, return_tensors="pt")["input_ids"].to("cuda")
    tokens_list = [_clean_token_for_display(t) for t in tokenizer.convert_ids_to_tokens(inputs[0].detach().cpu().tolist())] # Remove whitespace tokens
    with torch.no_grad():
        outputs = model(inputs, output_attentions=True)['attentions']
    attention_map = [attn_layer.detach().cpu() for attn_layer in outputs]

    attention_map = [attention_map[i][:, :, 1:, 1: ] for i in range(len(attention_map))]
    tokens_list = tokens_list[1:]

    selected_layers = attention_map
    selected_layer_offset = 0

    os.makedirs(_output_dir(), exist_ok=True)

    plot_attn_map_triangle(
        layers=selected_layers,
        layer_offset=selected_layer_offset,
        tokens_list=tokens_list,
        out_path=os.path.join(_output_dir(), "attention_map.png"),
    )
    plot_attn_map_last_row_individual_head(
        layers=selected_layers,
        layer_offset=selected_layer_offset,
        tokens_list=tokens_list,
        out_path=os.path.join(_output_dir(), "attention_last_row_heads.png"),
    )
    return attention_map


def view_attention_for_single_token(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    *,
    heatmap_vmin: float | None = None,
    heatmap_vmax: float | None = None,
    heatmap_clip_pct: float | None = None,
):
    """
    Extract a layers x heads heatmap for the option-letter tokens A and B
    (assuming the prompt begins with 'Answer with Option Letters: A.<...> and B.<...> ...').

    Saves:
    - outputs/label_view_A.png
    - outputs/label_view_B.png
    """
    inputs = tokenizer(question, return_tensors="pt")["input_ids"].to(model.device)
    tokens_list = [_clean_token_for_display(t) for t in tokenizer.convert_ids_to_tokens(inputs[0].detach().cpu().tolist())]

    with torch.no_grad():
        outputs = model(inputs, output_attentions=True)["attentions"]
    attention_map = [attn_layer.detach().cpu() for attn_layer in outputs]

    # Drop BOS to match prior visualizations
    attention_map = [layer[:, :, 1:, 1:] for layer in attention_map]
    tokens_list = tokens_list[1:]

    idx_a = _find_first_option_letter_token_index(tokens_list, "A")
    idx_b = _find_first_option_letter_token_index(tokens_list, "B")

    os.makedirs(_output_dir(), exist_ok=True)
    _plot_label_view_last_token_to_target(
        attention_map=attention_map,
        token_index=idx_a,
        tokens_list=tokens_list,
        out_path=os.path.join(_output_dir(), "label_view_A.png"),
        token_label="A",
        vmin=heatmap_vmin,
        vmax=heatmap_vmax,
        clip_pct=heatmap_clip_pct,
    )
    _plot_label_view_last_token_to_target(
        attention_map=attention_map,
        token_index=idx_b,
        tokens_list=tokens_list,
        out_path=os.path.join(_output_dir(), "label_view_B.png"),
        token_label="B",
        vmin=heatmap_vmin,
        vmax=heatmap_vmax,
        clip_pct=heatmap_clip_pct,
    )
    return {"A_index": idx_a, "B_index": idx_b, "tokens_list": tokens_list}

def extract_label_view_matrices(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
) -> dict:
    """
    Compute (layers x heads) attention matrices for labels A and B for `question`.
    Does not save any figures.
    """
    inputs = tokenizer(question, return_tensors="pt")["input_ids"].to(model.device)
    tokens_list = [_clean_token_for_display(t) for t in tokenizer.convert_ids_to_tokens(inputs[0].detach().cpu().tolist())]

    with torch.no_grad():
        outputs = model(inputs, output_attentions=True)["attentions"]
    attention_map = [attn_layer.detach().cpu() for attn_layer in outputs]

    # Drop BOS to match prior visualizations
    attention_map = [layer[:, :, 1:, 1:] for layer in attention_map]
    tokens_list = tokens_list[1:]

    idx_a = _find_first_option_letter_token_index(tokens_list, "A")
    idx_b = _find_first_option_letter_token_index(tokens_list, "B")

    data_a = _layers_heads_attention_last_token_to_target(attention_map, idx_a)
    data_b = _layers_heads_attention_last_token_to_target(attention_map, idx_b)

    return {
        "A_index": idx_a,
        "B_index": idx_b,
        "tokens_list": tokens_list,
        "data_A": data_a,
        "data_B": data_b,
    }



def extract_label_view_matrices_avg(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    *,
    num_runs: int = 10,
    root_seed: int = ROOT_SEED,
    topk_weights: int = 10,
) -> dict:
    """
    Run the same forward pass multiple times (with different RNG seeds) and return:
    - elementwise mean attention matrices (layers x heads) for A and B
    - mean/std entropy (bits) computed per run
    - mean/std top-k score computed per run
    """
    seeds = sample_run_seeds(root_seed, num_runs)

    data_a_runs: list[torch.Tensor] = []
    data_b_runs: list[torch.Tensor] = []
    entropy_a_runs: list[float] = []
    entropy_b_runs: list[float] = []
    score_a_runs: list[float] = []
    score_b_runs: list[float] = []

    first_info: dict | None = None
    for s in seeds:
        seed_everything(s)
        info = extract_label_view_matrices(model, tokenizer, question)
        if first_info is None:
            first_info = {k: v for k, v in info.items() if k not in ("data_A", "data_B")}

        da = info["data_A"]
        db = info["data_B"]
        data_a_runs.append(da)
        data_b_runs.append(db)

        entropy_a_runs.append(_entropy_bits_of_scores(da))
        entropy_b_runs.append(_entropy_bits_of_scores(db))
        score_a_runs.append(_topk_weight_ratio(da, topk_weights))
        score_b_runs.append(_topk_weight_ratio(db, topk_weights))

    if first_info is None:
        raise RuntimeError("num_runs produced no runs.")

    data_a_mean = torch.stack(data_a_runs, dim=0).mean(dim=0)
    data_b_mean = torch.stack(data_b_runs, dim=0).mean(dim=0)

    ent_a = torch.tensor(entropy_a_runs, dtype=torch.float32)
    ent_b = torch.tensor(entropy_b_runs, dtype=torch.float32)
    sc_a = torch.tensor(score_a_runs, dtype=torch.float32)
    sc_b = torch.tensor(score_b_runs, dtype=torch.float32)

    return {
        **first_info,
        "seeds": seeds,
        "num_runs": int(num_runs),
        "data_A": data_a_mean,
        "data_B": data_b_mean,
        "entropy_A_mean": float(ent_a.mean().item()),
        "entropy_A_std": float(ent_a.std(unbiased=False).item()),
        "entropy_B_mean": float(ent_b.mean().item()),
        "entropy_B_std": float(ent_b.std(unbiased=False).item()),
        "score_A_mean": float(sc_a.mean().item()),
        "score_A_std": float(sc_a.std(unbiased=False).item()),
        "score_B_mean": float(sc_b.mean().item()),
        "score_B_std": float(sc_b.std(unbiased=False).item()),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument(
        "--font_scale",
        type=float,
        default=2.0,
        help="Scale all plot text sizes (default doubles everything).",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=10,
        help="Number of runs to average for each prompt's attention/entropy.",
    )
    parser.add_argument(
        "--root_seed",
        type=int,
        default=ROOT_SEED,
        help="Root seed used to deterministically sample per-run seeds.",
    )
    parser.add_argument(
        "--heatmap_vmin",
        type=float,
        default=None,
        help="Fixed minimum for label-view heatmap color scaling (outputs/label_view_*.png).",
    )
    parser.add_argument(
        "--heatmap_vmax",
        type=float,
        default=None,
        help="Fixed maximum for label-view heatmap color scaling (outputs/label_view_*.png).",
    )
    parser.add_argument(
        "--heatmap_clip_pct",
        type=float,
        default=None,
        help="If set, use (p, 1-p) quantiles as (vmin, vmax) for a robust shared scale "
             "for label-view figures only. "
             "Must be in [0, 0.5). Ignored if --heatmap_vmin/--heatmap_vmax are set.",
    )
    parser.add_argument(
        "--topk_weights",
        type=int,
        default=10,
        help="K for Score = sum(Top K weights) / sum(All weights), shown on each subplot.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    set_plot_text_scale(args.font_scale)
    model, tokenizer = download_model(args.model_name)
    # attention_map = visualize_attention_map(model, tokenizer, prompt)
    prompt_specs = [
        ("No Keyword", prompt),
        ("Correct Keyword", washing_machine_prompt),
        ("Incorrect Keyword", kitchen_prompt),
    ]

    datas_a: list[torch.Tensor] = []
    datas_b: list[torch.Tensor] = []
    metrics_a: list[dict] = []
    metrics_b: list[dict] = []
    subplot_titles: list[str] = []

    for name, p in prompt_specs:
        info = extract_label_view_matrices_avg(
            model,
            tokenizer,
            p,
            num_runs=args.num_runs,
            root_seed=args.root_seed,
            topk_weights=args.topk_weights,
        )
        datas_a.append(info["data_A"])
        datas_b.append(info["data_B"])
        metrics_a.append(
            {
                "entropy_mean": info["entropy_A_mean"],
                "entropy_std": info["entropy_A_std"],
                "score_mean": info["score_A_mean"],
                "score_std": info["score_A_std"],
            }
        )
        metrics_b.append(
            {
                "entropy_mean": info["entropy_B_mean"],
                "entropy_std": info["entropy_B_std"],
                "score_mean": info["score_B_mean"],
                "score_std": info["score_B_std"],
            }
        )
        subplot_titles.append(name)
        print(
            {
                "prompt": name,
                "num_runs": info["num_runs"],
                "root_seed": args.root_seed,
                "seeds": info["seeds"],
                **{k: v for k, v in info.items() if k not in ("data_A", "data_B", "seeds", "num_runs")},
            }
        )

    os.makedirs(_output_dir(), exist_ok=True)
    plot_label_views_combined(
        label="A",
        datas=datas_a,
        subplot_titles=subplot_titles,
        out_path=os.path.join(_output_dir(), "label_view_A.png"),
        heatmap_vmin=args.heatmap_vmin,
        heatmap_vmax=args.heatmap_vmax,
        heatmap_clip_pct=args.heatmap_clip_pct,
        topk_weights=args.topk_weights,
        subplot_metrics=metrics_a,
    )

    # Also save the 3 split plots for label A.
    suffix_by_title = {
        "No Keyword": "no_keyword",
        "Correct Keyword": "correct",
        "Incorrect Keyword": "incorrect",
    }
    # Keep a consistent color scale across the split A plots (unless user overrides).
    vmin_a, vmax_a = _resolve_vmin_vmax(
        datas_a,
        vmin=args.heatmap_vmin,
        vmax=args.heatmap_vmax,
        clip_pct=args.heatmap_clip_pct,
    )
    if vmin_a is None and vmax_a is None:
        vmin_a = min(float(t.min().item()) for t in datas_a)
        vmax_a = max(float(t.max().item()) for t in datas_a)
        if vmin_a == vmax_a:
            eps = 1e-12
            vmin_a -= eps
            vmax_a += eps

    for title, data, m in zip(subplot_titles, datas_a, metrics_a):
        suffix = suffix_by_title.get(title, re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_"))
        plot_label_view_single(
            label="A",
            data=data,
            out_path=os.path.join(_output_dir(), f"label_view_A_{suffix}.png"),
            title=f"{title}: Attention(last token → 'A') across layers/heads",
            vmin=vmin_a,
            vmax=vmax_a,
            topk_weights=args.topk_weights,
            metrics=m,
        )
    plot_label_views_combined(
        label="B",
        datas=datas_b,
        subplot_titles=subplot_titles,
        out_path=os.path.join(_output_dir(), "label_view_B.png"),
        heatmap_vmin=args.heatmap_vmin,
        heatmap_vmax=args.heatmap_vmax,
        heatmap_clip_pct=args.heatmap_clip_pct,
        topk_weights=args.topk_weights,
        subplot_metrics=metrics_b,
    )

if __name__ == "__main__":
    main()

