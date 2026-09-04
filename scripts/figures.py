"""Figures for E0.1, E0.2, E0.3, E0.4 and E1. One simple figure per experiment."""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jspace.model import WORKSPACE_BAND

PROPS = ["language", "tense", "pos"]
C_A, C_B = "#B0641E", "#1F5FA8"   # (A) continue, (B) name
C_J, C_L = "#1F5FA8", "#9A9A9A"   # J-lens, logit lens
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": .25, "grid.linewidth": .5})


def boot(v, n=10000, seed=0):
    rng = np.random.default_rng(seed); v = np.asarray(v, float)
    d = rng.choice(v, size=(n, len(v)), replace=True).mean(1)
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


def band_span(ax):
    ax.axvspan(*WORKSPACE_BAND, color="#1F5FA8", alpha=.07, zorder=0)
    ax.text(np.mean(WORKSPACE_BAND), ax.get_ylim()[1], "workspace band 21–46",
            ha="center", va="top", fontsize=7, color="#1F5FA8")


# ---------------------------------------------------------------- E0.1
def fig_e0_1():
    r = json.load(open("results/e0_lens_sanity_27b.json"))
    d = r["E0.1a"]["boot_pos"]
    layers = sorted(int(l) for l in d["jlens_top5"])
    tg = tuple(d["targets"])
    hit = lambda top: [any(any(t in w.lower() for t in tg) for w in top[str(l)])
                       for l in layers]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7, 3.4),
                                  gridspec_kw={"height_ratios": [1.6, 1]})
    for row, (lab, top, c) in enumerate([("J-lens", d["jlens_top5"], C_J),
                                         ("logit lens", d["logit_lens_top5"], C_L)]):
        h = hit(top)
        ax.scatter([l for l, x in zip(layers, h) if x], [row] * sum(h), marker="s",
                   s=26, color=c, label=lab)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["J-lens", "logit lens"])
    ax.set_ylim(1.6, -0.6); ax.set_xlabel("layer")
    ax.set_title("E0.1a  ' Italy' in top-5 at the boot token, by layer\n"
                 "(two-hop prompt: the country shaped like a boot)", fontsize=9)
    for lab, top, c in [("J", d["jlens_top5"], C_J), ("LL", d["logit_lens_top5"], C_L)]:
        h = hit(top); first = next((l for l, x in zip(layers, h) if x), None)
        if first is not None:
            ax.annotate(f"first at L{first}", (first, 0 if lab == "J" else 1),
                        xytext=(6, -12 if lab == "J" else 12), textcoords="offset points",
                        fontsize=7.5, color=c)
    band_span(ax)

    ov = r["E0.1b"]["per_snippet_overlap"]
    ax2.bar(range(len(ov)), ov, color=C_J, width=.65)
    ax2.axhline(.6, color="k", ls="--", lw=.8)
    ax2.text(len(ov) - .5, .62, "0.60 required", ha="right", fontsize=7)
    ax2.axhline(.4, color="#B00", ls=":", lw=.8)
    ax2.text(len(ov) - .5, .42, "0.40 = K0", ha="right", fontsize=7, color="#B00")
    ax2.set_ylim(0, 1.05); ax2.set_xlabel("wikitext snippet")
    ax2.set_ylabel("top-5 overlap")
    ax2.set_title(f"E0.1b  J-lens vs logit-lens agreement at L{r['E0.1b']['layer']}"
                  f"  (mean {np.mean(ov):.2f})", fontsize=9)
    fig.tight_layout(); fig.savefig("figures/e0_1_lens_sanity.png"); plt.close(fig)


# ---------------------------------------------------------------- E0.2
def fig_e0_2():
    r = json.load(open("results/e0_difference_final.json"))["properties"]
    acc = [r[p]["E0.2"]["naming_accuracy"] for p in PROPS]
    n = [r[p]["E0.2"]["n_after_drop"] for p in PROPS]
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    bars = ax.bar(PROPS, acc, color=[C_B if a >= .9 else "#B0641E" for a in acc], width=.6)
    ax.axhline(.9, color="k", ls="--", lw=.8)
    ax.text(2.45, .91, "90% required", ha="right", fontsize=7)
    for bar, a, k in zip(bars, acc, n):
        ax.text(bar.get_x() + bar.get_width() / 2, a + .02, f"{a:.0%}\n({k} kept)",
                ha="center", fontsize=7.5)
    ax.set_ylim(0, 1.15); ax.set_ylabel("names the property correctly")
    ax.set_title("E0.2  behavioral positive control", fontsize=9)
    fig.tight_layout(); fig.savefig("figures/e0_2_behavioral.png"); plt.close(fig)


# ---------------------------------------------------------------- E0.3
def fig_e0_3():
    rng = json.load(open("results/e0_3_by_range.json"))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.1))
    ax = axes[0]
    keys = ["all layers", "band 21-46"]
    x = np.arange(len(PROPS)); w = .36
    for k, (key, c) in enumerate(zip(keys, [C_L, C_J])):
        a = [rng[p][key]["cells_A"] for p in PROPS]
        bvals = [rng[p][key]["cells_B"] for p in PROPS]
        ax.bar(x + (k - .5) * w, bvals, w * .48, color=c, label=f"(B) name — {key}")
        ax.bar(x + (k - .5) * w + w * .48, a, w * .48, color=c, alpha=.4,
               label=f"(A) continue — {key}")
    ax.set_xticks(x); ax.set_xticklabels(PROPS); ax.set_yscale("symlog")
    ax.set_ylabel("top-25 (layer, position) cells")
    ax.legend(fontsize=6.5, ncol=2); ax.set_title("E0.3  cell counts", fontsize=9)
    for i, p in enumerate(PROPS):
        ax.text(i, ax.get_ylim()[1], f"{rng[p]['band 21-46']['ratio']:.2f}×"
                if rng[p]["band 21-46"]["cells_A"] else "n/a",
                ha="center", va="top", fontsize=7.5)

    ax = axes[1]
    for p, c in zip(PROPS, [C_B, "#4C9F70", "#B0641E"]):
        z = np.load(f"results/e0_curves_{p}.npz"); L = z["layers"]
        diff = z["cells_by_layer_b"].mean(0) - z["cells_by_layer_a"].mean(0)
        ax.plot(L, diff, color=c, lw=1.4, label=p)
    ax.axhline(0, color="k", lw=.6)
    ax.set_xlabel("layer"); ax.set_ylabel("(B) − (A) top-25 cells per passage")
    ax.legend(fontsize=7); ax.set_title("E0.3  where the cells are", fontsize=9)
    band_span(ax)
    fig.tight_layout(); fig.savefig("figures/e0_3_cells.png"); plt.close(fig)


# ---------------------------------------------------------------- E0.4
def fig_e0_4():
    r = json.load(open("results/e0_difference_final.json"))
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    x = np.arange(len(PROPS)); w = .34
    stats = {}
    for k, (lab, keys, c) in enumerate([("J-lens", ("B", "A"), C_J),
                                        ("logit lens", ("B_ll", "A_ll"), C_L)]):
        means, errs, ls = [], [[], []], []
        for p in PROPS:
            z = np.load(f"results/e0_curves_{p}.npz"); L = list(z["layers"])
            l = r["properties"][p]["E1"]["l_star_in_band"]; i = L.index(l); ls.append(l)
            g = (z[keys[0]] - z[keys[1]])[:, i]
            lo, hi = boot(g); means.append(float(g.mean()))
            errs[0].append(g.mean() - lo); errs[1].append(hi - g.mean())
            stats.setdefault(p, {"l_star_in_band": l})[lab] = {
                "gap": float(g.mean()), "ci95": [float(lo), float(hi)]}
        ax.bar(x + (k - .5) * w, means, w, yerr=errs, color=c, label=lab,
               capsize=3, error_kw={"lw": .9})
    # Recompute the ratio from the data being plotted -- never annotate from a
    # stale file, since E0.2 filtering changed both n and l* for pos.
    for p in PROPS:
        gj = stats[p]["J-lens"]["gap"]; gl = stats[p]["logit lens"]["gap"]
        stats[p]["ratio"] = gj / gl if gl > 0 else float("inf")
        stats[p]["pass_1.5x"] = bool(gj > 0 and stats[p]["ratio"] >= 1.5)
    json.dump(stats, open("results/e0_4_in_band.json", "w"), indent=1)
    ax.axhline(0, color="k", lw=.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}\nl*={l}" for p, l in zip(PROPS, ls)])
    ax.set_ylabel("(B) − (A) coordinate gap at p*")
    ax.legend(fontsize=7.5)
    ax.set_title("E0.4  is it workspace content or output preparation?\n"
                 "J-lens gap must exceed the logit-lens gap by 1.5×", fontsize=9)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.22)
    for i, p in enumerate(PROPS):
        e = stats[p]; rr = e["ratio"]
        ax.text(i, ax.get_ylim()[1] * .93,
                ("∞" if rr == float("inf") else f"{rr:.2f}×") +
                ("  pass" if e["pass_1.5x"] else "  FAIL"),
                ha="center", fontsize=7.5,
                color="#1a7" if e["pass_1.5x"] else "#B00")
    fig.tight_layout(); fig.savefig("figures/e0_4_lens_vs_logitlens.png"); plt.close(fig)


# ---------------------------------------------------------------- E1
def fig_e1():
    r = json.load(open("results/e0_difference_final.json"))["properties"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.2), sharex=True)
    for j, p in enumerate(PROPS):
        z = np.load(f"results/e0_curves_{p}.npz"); L = z["layers"]
        n = z["A"].shape[0]
        ax = axes[0, j]
        for arr, lab, c in [(z["A"], "(A) predict next word", C_A),
                            (z["B"], "(B) identify property", C_B)]:
            m = arr.mean(0); se = arr.std(0, ddof=1) / np.sqrt(n)
            ax.plot(L, m, color=c, lw=1.4, label=lab)
            ax.fill_between(L, m - se, m + se, color=c, alpha=.22, lw=0)
        ls = r[p]["E1"]["l_star_in_band"]
        ax.axvline(ls, color="k", ls="--", lw=.9)
        ax.set_title(f"{p}   l* = {ls}   (n={n})", fontsize=9)
        if j == 0:
            ax.set_ylabel("J-lens coordinate at p*"); ax.legend(fontsize=6.5)
        band_span(ax)

        ax = axes[1, j]
        gap = z["B"] - z["A"]
        m = gap.mean(0); se = gap.std(0, ddof=1) / np.sqrt(n)
        zed = m / np.maximum(se, 1e-12)
        ax.plot(L, zed, color="#333", lw=1.2)
        ax.fill_between(L, 0, zed, where=zed > 2, color=C_B, alpha=.35, lw=0)
        ax.axhline(2, color="#B00", ls=":", lw=.9)
        ax.text(1, 2.25, "2 SE", fontsize=6.5, color="#B00")
        ax.axvline(ls, color="k", ls="--", lw=.9)
        ax.set_xlabel("layer")
        if j == 0:
            ax.set_ylabel("(B) − (A) gap  /  SE")
        band_span(ax)
    fig.suptitle("E1  where the fact enters: layer curves at p*, per-passage SE",
                 fontsize=10)
    fig.tight_layout(); fig.savefig("figures/e1_layer_curves.png"); plt.close(fig)


# ---------------------------------------------------------------- E2
def fig_e2(prop="language"):
    r = json.load(open(f"results/e2_attribution_{prop}.json"))
    fig = plt.figure(figsize=(11, 3.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.35], wspace=.42)
    kind_c = {"attn": C_B, "gdn": "#4C9F70", "mlp": "#B0641E"}

    # attribution mass -- the H1-vs-H2 answer
    ax = fig.add_subplot(gs[0])
    m = r["attribution_mass"]
    labs = ["attn", "gdn", "mlp"]
    stim = [m[f"{k}_at_stimulus"] for k in labs]
    ques = [m[f"{k}_at_question"] for k in labs]
    y = np.arange(3)
    ax.barh(y, stim, .55, color=[kind_c[k] for k in labs], label="at stimulus positions")
    ax.barh(y, ques, .55, left=stim, color=[kind_c[k] for k in labs], alpha=.4,
            label="at question positions")
    for i, (a, b_) in enumerate(zip(stim, ques)):
        ax.text(a + b_ + .01, i, f"{a:.0%} + {b_:.0%}", va="center", fontsize=7)
    ax.set_yticks(y); ax.set_yticklabels(labs); ax.set_xlim(0, .45)
    ax.set_xlabel("share of |attribution| mass")
    ax.legend(fontsize=6.5, loc="lower right")
    ax.set_title("E2  attribution mass\n(H1: attention?  H2: MLP?)", fontsize=9)

    # attribution vs true patching effect -- the K3 diagnostic
    ax = fig.add_subplot(gs[1])
    est = [d["attribution_estimate"] for d in r["all_candidates"]]
    true = [d["true_delta_m"] for d in r["all_candidates"]]
    kinds = [d["node"].split("/")[0] for d in r["all_candidates"]]
    for k in labs:
        xs = [e for e, kk in zip(est, kinds) if kk == k]
        ys = [t for t, kk in zip(true, kinds) if kk == k]
        ax.scatter(xs, ys, s=16, color=kind_c[k], label=k, alpha=.85, edgecolor="none")
    lim = [min(est + true) * 1.1, max(est + true + [0]) * 1.1]
    ax.plot(lim, lim, color="k", lw=.7, ls="--")
    ax.set_xlabel("attribution estimate"); ax.set_ylabel("true Δm (activation patching)")
    ax.legend(fontsize=6.5)
    ax.set_title(f"E2  attribution is reliable here\nr = {r['attribution_vs_patching_corr']:.3f}"
                 f"  (K3 fires below 0.4)", fontsize=9)

    # top verified promoters
    ax = fig.add_subplot(gs[2])
    top = r["verified_nodes"][:14][::-1]
    y = np.arange(len(top))
    ax.barh(y, [-d["true_delta_m"] for d in top],
            xerr=[d["true_delta_se"] for d in top],
            color=[kind_c[d["kind"]] for d in top], height=.68,
            error_kw={"lw": .8, "ecolor": "#444"})
    ax.set_yticks(y); ax.set_yticklabels([d["node"] for d in top], fontsize=7)
    ax.set_xlabel("drop in J-lens coordinate when ablated  (−Δm)")
    ax.axvline(r["BA_gap_mean"], color="k", ls=":", lw=.9)
    ax.text(r["BA_gap_mean"], len(top) - .4, f" full (B)−(A) gap = {r['BA_gap_mean']:.2f}",
            fontsize=6.5, va="top")
    ax.set_title(f"E2  verified promoters, {prop}  (l*={r['l_star']})", fontsize=9)
    fig.savefig(f"figures/e2_attribution_{prop}.png", bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- E3
def fig_e3():
    import os
    props = [p for p in PROPS if os.path.exists(f"results/e3_causal_{p}.json")]
    if not props:
        return
    fig, axes = plt.subplots(1, len(props), figsize=(3.7 * len(props), 3.2), squeeze=False)
    for ax, p in zip(axes[0], props):
        r = json.load(open(f"results/e3_causal_{p}.json"))
        ks = [c["k"] for c in r["curve"]]
        fr = [c["frac_of_gap_closed"] for c in r["curve"]]
        ax.plot(ks, fr, "o-", color=C_B, lw=1.5, ms=4, label="P_prop (E2 order)")
        for kind, c, lab in [("matched", "#888", "random, size+kind matched"),
                             ("broadcast_band", "#B0641E", "random broadcast-band heads")]:
            bl = r["random_baselines"][kind]
            ax.errorbar([len(r["P_prop"])], [bl["mean"]], yerr=[bl["sd"]], fmt="s",
                        color=c, ms=5, capsize=3, label=lab)
        ax.axhline(.70, color="k", ls="--", lw=.8)
        ax.text(ks[0], .72, "70% required", fontsize=6.5)
        ax.axhline(0, color="k", lw=.6)
        ax.set_xlabel("nodes ablated (k)"); ax.set_title(
            f"{p}  |P|={r['size']}  →  {r['frac_of_gap_closed']:.0%}", fontsize=9)
        if p == props[0]:
            ax.set_ylabel("fraction of the (B)−(A) gap closed")
            ax.legend(fontsize=6, loc="lower right")
        ax.set_ylim(-.35, 1.05)
    fig.suptitle("E3  counterfactual ablation: does a small set suffice?", fontsize=10)
    fig.tight_layout(); fig.savefig("figures/e3_causal.png"); plt.close(fig)


# ---------------------------------------------------------------- E4
def fig_e4():
    import os
    if not os.path.exists("results/e4_cross_ablation.json"):
        return
    r = json.load(open("results/e4_cross_ablation.json"))
    M = np.array([[r["matrix"][j][i] for i in PROPS] for j in PROPS])
    vac = np.array([[r["detail"][j][i]["n_applied"] == 0 for i in PROPS] for j in PROPS])
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    im = ax.imshow(np.clip(M, -.5, 1.5), cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    for a in range(3):
        for b_ in range(3):
            txt = "n/a" if vac[a, b_] else f"{M[a, b_]:.2f}"
            ax.text(b_, a, txt, ha="center", va="center", fontsize=9,
                    color="k" if abs(M[a, b_]) < .9 else "w")
    ax.set_xticks(range(3)); ax.set_xticklabels([f"ablate\nP_{p}" for p in PROPS], fontsize=8)
    ax.set_yticks(range(3)); ax.set_yticklabels([f"measure\n{p}" for p in PROPS], fontsize=8)
    ax.set_title("E4  cross-ablation transfer\n"
                 "1.0 = same effect as the property's own set", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=.046, label="fraction of within-property drop")
    ax.text(0, 2.75, "n/a = every node of P_pos sits above language's l*=24,\n"
                     "so the cell is vacuous by construction, not a null result",
            fontsize=6.5, ha="left", transform=ax.transData)
    fig.tight_layout(); fig.savefig("figures/e4_cross_ablation.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------- pivot experiments
def fig_pivot():
    import os
    ok = all(os.path.exists(f) for f in
             ["results/e5_granularity.json", "results/e6_direct_effect_language.json",
              "results/e7_boundary_language.json", "results/e8_reentry_language.npz"])
    if not ok:
        return
    g = json.load(open("results/e5_granularity.json"))["language"]
    d = json.load(open("results/e6_direct_effect_language.json"))
    e7 = json.load(open("results/e7_boundary_language.json"))
    z = np.load("results/e8_reentry_language.npz")
    kind_c = {"attn": C_B, "gdn": "#4C9F70", "mlp": "#B0641E"}

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6))

    # (a) granularity
    ax = axes[0, 0]
    labs = ["attn", "gdn", "mlp"]
    x = np.arange(3); w = .36
    old = [g["mass_per_head_E2"][f"{k}_at_stimulus"] for k in labs]
    new = [g["mass_block_level"][f"{k}_at_stimulus"] for k in labs]
    ax.bar(x - w/2, old, w, color="#BBB", label="per-head (unmatched)")
    ax.bar(x + w/2, new, w, color=[kind_c[k] for k in labs], label="block level (matched)")
    ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylabel("share of mass @ stimulus")
    ax.legend(fontsize=7); ax.set_title("(a) granularity fix: attention shrinks", fontsize=9)

    # (b) direct vs indirect
    ax = axes[0, 1]
    rows = d["part_A_direct_split"]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [-r["direct"] for r in rows], color="#444", label="direct (writes at l*)")
    ax.barh(y, [-r["indirect"] for r in rows], left=[-r["direct"] for r in rows],
            color="#7FB3D5", label="indirect (via later layers)")
    ax.set_yticks(y); ax.set_yticklabels([r["block"] for r in rows], fontsize=7)
    ax.axvline(0, color="k", lw=.6); ax.legend(fontsize=7)
    ax.set_xlabel("−Δm"); ax.set_title(
        f"(b) direct/indirect split (median {d['verdict']['median_direct_fraction']:.0%} direct)",
        fontsize=9)

    # (c) boundary carrying
    ax = axes[1, 0]
    ls = [r["layer"] for r in e7["per_layer"]]
    fr = [r["frac_of_gap_killed"] for r in e7["per_layer"]]
    cs = [kind_c[r["kind"]] for r in e7["per_layer"]]
    ax.bar(ls, fr, color=cs, width=.75)
    ax.axhline(0, color="k", lw=.6)
    for r in e7["per_layer"]:
        if r["frac_of_gap_killed"] > .09:
            ax.text(r["layer"], r["frac_of_gap_killed"] + .01, f"L{r['layer']}",
                    ha="center", fontsize=6.5)
    ax.set_xlabel("layer"); ax.set_ylabel("fraction of the gap killed")
    ax.set_title("(c) which layer carries the instruction across the\n"
                 "question→stimulus boundary (green = GDN, blue = attention)", fontsize=9)

    # (d) re-entry
    ax = axes[1, 1]
    layers = z["layers"]; ps = int(z["p_star"])
    C, A, R = z["clean"][:, ps], z["ablated"][:, ps], z["condA"][:, ps]
    ax.plot(layers, C, color=C_B, lw=1.5, label="(B) clean")
    ax.plot(layers, A, color="#C0392B", lw=1.5, label="(B) with P ablated")
    ax.plot(layers, R, color=C_A, lw=1.2, ls="--", label="(A) baseline")
    ax.axvline(24, color="k", ls="--", lw=.9); ax.text(24.4, ax.get_ylim()[1]*.95, "l*", fontsize=7)
    ax.axvline(32, color="#C0392B", ls=":", lw=1.1)
    ax.text(32.4, ax.get_ylim()[1]*.85, "re-entry\nL32", fontsize=7, color="#C0392B")
    ax.set_xlabel("layer"); ax.set_ylabel("Spanish coordinate at p*")
    ax.legend(fontsize=7); ax.set_title("(d) re-entry: removed at l*, restored by L32", fontsize=9)

    fig.suptitle("Pivot experiments: granularity, direct effect, boundary crossing, re-entry",
                 fontsize=10)
    fig.tight_layout(); fig.savefig("figures/pivot_experiments.png"); plt.close(fig)


# ---------------------------------------------------------------- E9
def fig_e9():
    import os
    if not os.path.exists("results/e9_carrier_specificity.json"):
        return
    r = json.load(open("results/e9_carrier_specificity.json"))
    m, gap = r["unpatched_coord"], r["gap"]
    rows = r["per_layer"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.7),
                                  gridspec_kw={"width_ratios": [1, 1.9]})

    # (a) the control: what each question alone does to the Spanish coordinate
    order = ["A", "tense", "pos", "language"]
    labs = ["(A)\npredict\nnext word", "identify\nthe tense",
            "identify the\npart of speech", "identify\nthe language"]
    cols = [C_A, "#4C9F70", "#B0641E", C_B]
    vals = [m[k] for k in order]
    ax.bar(range(4), vals, color=cols, width=.66)
    ax.set_ylim(min(vals) - .06, max(vals) + .09)
    ax.axhline(m["A"], color=C_A, ls=":", lw=1)
    ax.axhline(m["language"], color=C_B, ls=":", lw=1)
    for i, k in enumerate(order):
        ax.text(i, vals[i] + .012, f"{(m[k] - m['A']) / gap:.0%}", ha="center", fontsize=8)
    ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=7)
    ax.set_ylabel("Spanish coordinate at (L24, p*)")
    ax.set_title("(a) same Spanish passage, four questions\n"
                 "% = share of the language-vs-(A) gap", fontsize=9)

    # (b) boundary patch by source
    x = np.array([r_["layer"] for r_ in rows]); w = .27
    # Distinct colours per source: an earlier version gave "from (A)" and
    # "from pos" the same hue, making two series indistinguishable.
    for k, (src, c, lab) in enumerate([("A", "#33383D", "from (A) predict-next-word"),
                                       ("tense", "#4C9F70", "from tense question"),
                                       ("pos", "#D08A2E", "from part-of-speech question")]):
        ax2.bar(x + (k - 1) * w, [r_[src] for r_ in rows], w, color=c, label=lab)
    ax2.scatter(x, [r_["language"] for r_ in rows], s=9, facecolors="none",
                edgecolors="#C0392B", lw=.8, zorder=5,
                label="identity control (= 0 everywhere)")
    ax2.axhline(0, color="k", lw=.6)
    l9 = next(r_ for r_ in rows if r_["layer"] == 9)
    ax2.annotate("L9 carries the most from (A) — but the\n"
                 "tense question's carry substitutes for free",
                 xy=(9 - w, l9["A"]), xytext=(12.5, .235), fontsize=7,
                 arrowprops=dict(arrowstyle="->", lw=.8))
    ax2.annotate("", xy=(9, l9["tense"]), xytext=(12.4, .225),
                 arrowprops=dict(arrowstyle="->", lw=.8))
    ax2.set_xlabel("layer whose boundary state is patched")
    ax2.set_ylabel("fraction of the (B)−(A) gap killed")
    ax2.set_ylim(-.26, .34)
    ax2.legend(fontsize=6.5, ncol=2, loc="lower left")
    ax2.set_title("(b) is the carried signal generic or property-specific?\n"
                  "high = that source fails to carry it; ~0 = interchangeable", fontsize=9)
    fig.tight_layout(); fig.savefig("figures/e9_carrier_specificity.png"); plt.close(fig)


# --------------------------------------------------------------- E10
def fig_e10():
    import os
    if not os.path.exists("results/e10_contentless.json"):
        return
    r = json.load(open("results/e10_contentless.json"))
    m, gap, rows = r["unpatched_coord"], r["gap"], r["per_layer"]
    LING = ["tense", "pos", "language"]
    CONT = ["wordcount", "firstletter", "linecount", "linewidth"]
    NICE = {"A": "(A) predict\nnext word", "tense": "tense", "pos": "part of\nspeech",
            "language": "language", "wordcount": "how many\nwords",
            "firstletter": "first\nletter", "linecount": "how many\nlines",
            "linewidth": "longest\nline width"}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.9),
                                  gridspec_kw={"width_ratios": [1.25, 1]})

    order = ["A"] + LING + CONT
    share = [(m[k] - m["A"]) / gap for k in order]
    cols = ["#999"] + [C_B] * 3 + ["#B0641E"] * 4
    ax.bar(range(len(order)), share, color=cols, width=.68)
    ax.axhline(0, color="k", lw=.7); ax.axhline(1, color=C_B, ls=":", lw=1)
    for i, v in enumerate(share):
        ax.text(i, v + (.06 if v >= 0 else -.13), f"{v:.0%}", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([NICE[k] for k in order], fontsize=7)
    ax.set_ylabel("Spanish coordinate at (L24, p*)\nas a share of the language-vs-(A) gap")
    ax.set_ylim(-1.25, 2.15)
    ax.text(2, 1.9, "linguistic questions", color=C_B, ha="center", fontsize=8)
    ax.text(5.5, -1.15, "contentless questions", color="#B0641E", ha="center", fontsize=8)
    ax.set_title("(a) only *linguistic* questions admit Spanish to the workspace.\n"
                 "Counting words or letters pushes it BELOW the no-question baseline.",
                 fontsize=9)

    l9 = next(x for x in rows if x["layer"] == 9)
    srcs = LING[:2] + CONT
    vals = [l9[s_] for s_ in srcs]
    ax2.bar(range(len(srcs)), vals,
            color=[C_B, C_B] + ["#B0641E"] * 4, width=.6)
    ax2.axhline(l9["A"], color="#33383D", ls="--", lw=1.1)
    ax2.text(len(srcs) - .5, l9["A"] + .012, "patching from (A) = 28.0%",
             ha="right", fontsize=7)
    ax2.axhline(0, color="k", lw=.7)
    ax2.set_xticks(range(len(srcs)))
    ax2.set_xticklabels([NICE[s_] for s_ in srcs], fontsize=7)
    ax2.set_ylabel("fraction of the (B)−(A) gap killed")
    ax2.set_title("(b) L9 boundary patch: a linguistic question's carry\n"
                  "substitutes for free; a contentless one does not", fontsize=9)
    fig.tight_layout(); fig.savefig("figures/e10_contentless.png"); plt.close(fig)


# --------------------------------------------------------------- E11
def fig_e11():
    import os
    if not os.path.exists("results/e11_arms.json"):
        return
    r = json.load(open("results/e11_arms.json"))
    M, arms = r["matrix"], r["arms"]
    cols = ["Spanish", "past", "formal", "informal", "English", "table",
            "purple", "Thursday", "adjective", "noun"]
    nice_arm = {"none": "passage only*", "neutral": "neutral instruction",
                "A": "(A) predict next word", "linewidth": "longest line width",
                "wordcount": "how many words", "register": "register",
                "tense": "tense", "pos": "part of speech", "language": "language"}

    fig = plt.figure(figsize=(12.4, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.5], wspace=.3)

    # (a) the ladder, paired, with SE
    ax = fig.add_subplot(gs[0])
    order = ["none", "neutral", "A", "linewidth", "wordcount", "register",
             "tense", "language", "pos"]
    d = [M[a]["Spanish"]["delta"] for a in order]
    e = [M[a]["Spanish"]["se"] for a in order]
    col = []
    for a in order:
        t = M[a]["Spanish"]["delta"] / M[a]["Spanish"]["se"] if M[a]["Spanish"]["se"] else 0
        col.append(C_B if t >= 2 else ("#BBB" if a != "A" else "#666"))
    ax.barh(range(len(order)), d, xerr=e, color=col, height=.66,
            error_kw={"lw": .9, "ecolor": "#333"})
    ax.axvline(0, color="k", lw=.7)
    ax.axvline(M["language"]["Spanish"]["delta"], color=C_B, ls=":", lw=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([nice_arm[a] for a in order], fontsize=7.5)
    ax.invert_yaxis()
    for i, a in enumerate(order):
        t = M[a]["Spanish"]["delta"] / M[a]["Spanish"]["se"] if M[a]["Spanish"]["se"] else 0
        ax.text(max(d[i], 0) + e[i] + .02, i, f"t={t:.1f}", va="center", fontsize=6.5,
                color="#222" if abs(t) >= 2 else "#888")
    ax.set_xlabel("Δ Spanish coordinate vs (A), paired over 30 passages")
    ax.set_title("(a) blue = significant (|t| ≥ 2). Contentless questions and a\n"
                 "neutral instruction leave Spanish at the (A) level.", fontsize=9)

    # (b) the matrix
    ax = fig.add_subplot(gs[1])
    data = np.array([[M[a][c]["delta"] for c in cols] +
                     [M[a]["_random_null"]["mean"]] for a in arms])
    im = ax.imshow(data, cmap="RdBu_r", vmin=-.5, vmax=.5, aspect="auto")
    for i in range(len(arms)):
        for j in range(len(cols) + 1):
            v = data[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="w" if abs(v) > .3 else "k")
    ax.set_xticks(range(len(cols) + 1))
    ax.set_xticklabels(cols + ["random\n(24 tok)"], fontsize=6.5, rotation=45, ha="right")
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([nice_arm[a] for a in arms], fontsize=7)
    ax.set_title("(b) every passage-global property rises together under a grammatical\n"
                 "question — and ' adjective', the answer to the POS question, does not",
                 fontsize=9)
    fig.colorbar(im, ax=ax, fraction=.03, label="Δ vs (A)")
    fig.text(.01, .01, "* passage-only arm has no prefix, so its absolute position "
                       "differs from every other arm; shown for reference only.",
             fontsize=6.5, color="#666")
    fig.savefig("figures/e11_arms_matrix.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import os, sys
    for f in (fig_e0_1, fig_e0_2, fig_e0_3, fig_e0_4, fig_e1):
        f(); print("ok", f.__name__)
    for prop in PROPS:
        if os.path.exists(f"results/e2_attribution_{prop}.json"):
            fig_e2(prop); print("ok fig_e2", prop)
    fig_e3(); print("ok fig_e3")
    fig_e4(); print("ok fig_e4")
    fig_pivot(); print("ok fig_pivot")
    fig_e9(); print("ok fig_e9")
    fig_e11(); print("ok fig_e11")
