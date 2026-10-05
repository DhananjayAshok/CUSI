# TOY (skill_discovery.md §1.5): throwaway first-build scaffolding, not part of the kept code.
# Strip it by deleting cusi_search/toy/, run_preexplore_toy.py, slurm/preexplore_toy.sh,
# scripts/slurm/preexplore_toy.sh, tests/*toy* and the TOY block in skill_discovery.md §1.5.
# Nothing outside those files imports this.
"""Toy static HTML viewer for one search run directory (tree_io format):

    python -m cusi_search.toy.viewer --run_dir <out>      # writes <out>/index.html

Left: the tree (thumbnail, depth, Q, N, P per node). Click a node: its frame, stats, the VLM
prior's score and reasoning, the segment that reached it (frames + actions + scores), and the
expansions started from it. Bottom: per iteration, the chosen node's probability, the entropy of
the selection distribution, and the tree / cell counts; plus the selection log.
"""
import json
import os
import click
from cusi_search.toy.tree_io import read_jsonl

PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Search Tree Viewer</title>
<style>
:root { --bg:#fafafa; --fg:#1d1d1f; --muted:#6b6b70; --line:#d9d9de; --card:#ffffff; --accent:#2f6fde; --hi:#e8f0fd; }
@media (prefers-color-scheme: dark) { :root { --bg:#16171a; --fg:#e8e8ea; --muted:#9a9aa2; --line:#33343a; --card:#1f2024; --accent:#7aa7ff; --hi:#24324d; } }
* { box-sizing:border-box; } body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.4 system-ui,sans-serif; }
header { padding:12px 16px; border-bottom:1px solid var(--line); } header h1 { font-size:17px; margin:0 0 4px; }
.muted { color:var(--muted); } main { display:flex; gap:12px; padding:12px 16px; flex-wrap:wrap; }
#tree { flex:1 1 340px; max-height:70vh; overflow:auto; background:var(--card); border:1px solid var(--line); border-radius:8px; padding:8px; }
#detail { flex:2 1 480px; background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px; max-height:70vh; overflow:auto; }
ul { list-style:none; margin:0; padding-left:16px; } li > .n { display:flex; gap:6px; align-items:center; padding:2px 4px; border-radius:6px; cursor:pointer; }
li > .n:hover, li > .n.sel { background:var(--hi); } .n img { height:32px; image-rendering:pixelated; border:1px solid var(--line); }
.frames { display:flex; flex-wrap:wrap; gap:8px; } .frames figure { margin:0; width:150px; font-size:12px; }
.frames img { width:150px; image-rendering:pixelated; border:1px solid var(--line); } .big { max-width:100%; max-height:300px; image-rendering:pixelated; border:1px solid var(--line); }
table { border-collapse:collapse; font-size:12px; } td, th { border-bottom:1px solid var(--line); padding:2px 6px; text-align:left; }
section.bottom { padding:0 16px 16px; } svg { width:100%; height:180px; background:var(--card); border:1px solid var(--line); border-radius:8px; }
.legend span { margin-right:12px; font-size:12px; } .tag { font-size:11px; padding:0 4px; border-radius:4px; border:1px solid var(--line); }
pre { white-space:pre-wrap; font-size:12px; background:var(--bg); padding:6px; border-radius:6px; max-height:160px; overflow:auto; }
</style></head><body>
<header><h1>Search tree: __TITLE__</h1><div class="muted" id="summary"></div></header>
<main><div id="tree"></div><div id="detail"><p class="muted">Click a node.</p></div></main>
<section class="bottom"><h3>Selection over time</h3>
<div class="legend"><span style="color:#2f6fde">■ chosen p</span><span style="color:#d9822b">■ entropy / log(#candidates)</span><span style="color:#3a9d5d">■ nodes / max</span><span style="color:#a64ca6">■ cells / max</span></div>
<svg id="chart" viewBox="0 0 1000 180" preserveAspectRatio="none"></svg>
<div style="overflow:auto;max-height:240px"><table id="sel"></table></div></section>
<script>
const D = __DATA__;
const nodes = D.nodes, exps = D.expansions, sel = D.selection;
const byId = Object.fromEntries(nodes.map(n => [n.id, n]));
const f3 = x => (x === null || x === undefined) ? "-" : (+x).toFixed(3);
document.getElementById("summary").textContent = Object.entries(D.summary).map(([k, v]) => k + ": " + (typeof v === "number" ? +v.toFixed(2) : v)).join(" · ");
function li(n) {
  const kids = n.children.map(c => li(byId[c])).join("");
  const flags = Object.keys(n.flags || {}).map(f => `<span class="tag">${f}</span>`).join(" ");
  return `<li><div class="n" data-id="${n.id}"><img src="${n.frame}" loading="lazy"><span>#${n.id} <span class="muted">d${n.depth} c${n.cell} · exp ${n.n_expanded} · Q ${f3(n.q)} · N ${n.n} · P ${f3(n.prior)}</span> ${flags}</span></div>${kids ? "<ul>" + kids + "</ul>" : ""}</li>`;
}
document.getElementById("tree").innerHTML = "<ul style='padding-left:0'>" + nodes.filter(n => n.parent === null).map(li).join("") + "</ul>";
function stepFig(s) { return `<figure><img src="${s.frame}" loading="lazy"><figcaption>t${s.t} · v ${f3(s.value)}${s.new_cell ? " · new cell" : ""}${s.node_id !== null ? " · node #" + s.node_id : ""}${s.info && s.info.valid === false ? " · invalid" : ""}<br><code>${(s.action || "").replace(/</g, "&lt;").slice(-120)}</code></figcaption></figure>`; }
function show(id) {
  document.querySelectorAll(".n").forEach(e => e.classList.toggle("sel", +e.dataset.id === id));
  const n = byId[id];
  let h = `<h3>Node #${n.id} <span class="muted">(${n.env_key}, depth ${n.depth}, cell ${n.cell})</span></h3><img class="big" src="${n.frame}">`;
  h += `<table><tr><th>expanded</th><td>${n.n_expanded}</td><th>Q</th><td>${f3(n.q)}</td><th>N</th><td>${n.n}</td><th>P</th><td>${f3(n.prior)}</td><th>created at iter</th><td>${n.created_at}</td></tr></table>`;
  if (n.prior_reason || n.prior_raw) h += `<p><b>Prior:</b> ${n.prior_reason || "<i>(no reason parsed)</i>"}</p><pre>${(n.prior_raw || "").replace(/</g, "&lt;")}</pre>`;
  if (n.segment) { const [e, a, b] = n.segment; h += `<h4>Segment that reached it (expansion ${e}, steps ${a}-${b})</h4><div class="frames">${exps[e].steps.slice(a, b + 1).map(stepFig).join("")}</div>`; }
  const mine = exps.filter(e => e.node_id === id);
  mine.forEach(e => { h += `<h4>Expansion ${e.id} (iteration ${e.iteration}): yield ${f3(e.yield)}, children ${e.children.join(", ") || "none"}, valid ${f3(e.stats.valid_rate)}</h4><div class="frames">${e.steps.map(stepFig).join("")}</div>`; });
  if (Object.keys(n.texts || {}).length) h += `<h4>Texts</h4><pre>${Object.values(n.texts).join("\n").replace(/</g, "&lt;")}</pre>`;
  document.getElementById("detail").innerHTML = h;
}
document.querySelectorAll(".n").forEach(e => e.addEventListener("click", () => show(+e.dataset.id)));
if (nodes.length) show(0);
// chart
const W = 1000, H = 180, T = Math.max(1, sel.length - 1);
const maxN = Math.max(1, ...sel.map(r => r.n_nodes || 0)), maxC = Math.max(1, ...sel.map(r => r.n_cells || 0));
function line(vals, color) { const pts = vals.map((v, i) => `${(i / T) * W},${H - 8 - v * (H - 16)}`).join(" "); return `<polyline fill="none" stroke="${color}" stroke-width="2" points="${pts}"/>`; }
const ent = sel.map(r => { const p = r.probs || []; if (p.length < 2) return 0; return -p.reduce((a, x) => a + (x > 0 ? x * Math.log(x) : 0), 0) / Math.log(p.length); });
document.getElementById("chart").innerHTML = sel.length ? line(sel.map(r => r.prob || 0), "#2f6fde") + line(ent, "#d9822b") + line(sel.map(r => (r.n_nodes || 0) / maxN), "#3a9d5d") + line(sel.map(r => (r.n_cells || 0) / maxC), "#a64ca6") : "";
document.getElementById("sel").innerHTML = "<tr><th>iter</th><th>node</th><th>E</th><th>p</th><th>#cand</th><th>steps</th><th>children</th><th>yield</th><th>new cells</th><th>valid</th><th>s</th></tr>" +
  sel.map(r => `<tr><td>${r.iteration}</td><td><a href="#" onclick="show(${r.selected});return false">#${r.selected}</a></td><td>${f3(r.energy)}</td><td>${f3(r.prob)}</td><td>${(r.candidates || []).length}</td><td>${r.steps}</td><td>${(r.children || []).join(",")}</td><td>${f3(r.yield)}</td><td>${r.new_cells}</td><td>${f3(r.valid_rate)}</td><td>${f3(r.seconds)}</td></tr>`).join("");
</script></body></html>
"""


def write_viewer(*, run_dir: str) -> str:
    data = {"nodes": read_jsonl(os.path.join(run_dir, "nodes.jsonl")),
            "expansions": read_jsonl(os.path.join(run_dir, "expansions.jsonl")),
            "selection": read_jsonl(os.path.join(run_dir, "selection.jsonl")),
            "summary": json.load(open(os.path.join(run_dir, "summary.json")))}
    # nodes.jsonl lists every node; the page needs expansions indexed by id
    data["expansions"] = sorted(data["expansions"], key=lambda e: e["id"])
    html = PAGE.replace("__DATA__", json.dumps(data).replace("</", "<\\/")).replace(
        "__TITLE__", os.path.basename(os.path.normpath(run_dir)))
    path = os.path.join(run_dir, "index.html")
    with open(path, "w") as f:
        f.write(html)
    return path


@click.command()
@click.option("--run_dir", required=True)
def main(run_dir):
    print(write_viewer(run_dir=run_dir))


if __name__ == "__main__":
    main()
