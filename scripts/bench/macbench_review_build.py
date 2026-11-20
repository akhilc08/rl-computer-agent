"""Build the local human-review page for a sample of MacBench commands.

Samples --n commands stratified by target kind (ax/ocr/icon/menubar/none) and spread over apps, and writes
<bench>/review/{index.html, items.js}. The page reads ../images/<screen>.jpg, keeps verdicts in the browser's
localStorage and exports them as JSON; nothing leaves the machine. Score the export with macbench_review_merge.py.

  .venv/bin/python scripts/bench/macbench_review_build.py --bench data/macbench/v1 --n 300
  open data/macbench/v1/review/index.html
"""
import argparse
import collections
import json
import random
from pathlib import Path



def kind(c):
    return "none" if c["absent"] or not c["targets"] else c["target_kind"]


def sample(cmds, n, rng):
    by_kind = collections.defaultdict(lambda: collections.defaultdict(list))
    for c in cmds:
        by_kind[kind(c)][c["app"]].append(c)
    # Equal share per kind (small kinds give all they have, the rest goes to the others), round-robin over apps.
    quota, left = {}, n
    ks = sorted(by_kind, key=lambda k: sum(map(len, by_kind[k].values())))
    for i, k in enumerate(ks):
        have = sum(map(len, by_kind[k].values()))
        quota[k] = min(have, left // (len(ks) - i))
        left -= quota[k]
    out = []
    for k in ks:
        apps = {a: rng.sample(v, len(v)) for a, v in by_kind[k].items()}
        order = sorted(apps)
        rng.shuffle(order)
        got = []
        while len(got) < quota[k]:
            for a in order:
                if apps[a] and len(got) < quota[k]:
                    got.append(apps[a].pop())
        out += got
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ids", help="review exactly these command ids (one per line, in this order) instead of a sample")
    ap.add_argument("--out_name", default="review", help="folder under the bench for the page")
    args = ap.parse_args()
    bench = Path(args.bench)
    screens = {s["id"]: s for s in map(json.loads, open(bench / "screens.jsonl"))}
    cmds = [json.loads(l) for l in open(bench / "commands.jsonl")]
    if args.ids:
        by_id = {c["id"]: c for c in cmds}
        picked = [by_id[i] for i in open(args.ids).read().split()]
    else:
        picked = sample(cmds, args.n, random.Random(args.seed))
    items = []
    for c in picked:
        s = screens[c["screen"]]
        items.append({"id": c["id"], "app": c["app"], "kind": kind(c), "command": c["command"], "style": c["style"],
                      "action": c["action"], "text": c["text"], "text_source": c["text_source"],
                      "absent": c["absent"], "ambiguous": c["ambiguous"], "irreversible": c["irreversible"],
                      "image": "../" + s["image"], "size": s["img_size"], "title": s.get("title"),
                      "targets": [{"id": t["id"], "role": t["role"], "label": t["label"], "box": t.get("box_px")}
                                  for t in c["target_boxes"]]})
    out = bench / args.out_name
    out.mkdir(exist_ok=True)
    (out / "items.js").write_text("window.ITEMS = " + json.dumps(items) + ";\n")
    (out / "index.html").write_text(PAGE.replace("__KEY__", f"macbench-{args.out_name}-{bench.name}"))
    print(len(items), "items", dict(collections.Counter(i["kind"] for i in items)),
          "apps", len({i["app"] for i in items}), "->", out / "index.html")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>MacBench Review</title>
<style>
:root{--bg:#f6f6f4;--fg:#1d1d1f;--mute:#6b6b70;--card:#fff;--line:#ddd;--hit:#e5322d;--ok:#1f8a4c;--bad:#c4332b;--warn:#b7791f}
@media (prefers-color-scheme:dark){:root{--bg:#18181a;--fg:#ececee;--mute:#9a9aa0;--card:#232326;--line:#38383c}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,system-ui,sans-serif}
header{display:flex;gap:16px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line);flex-wrap:wrap}
header b{font-size:16px}#prog{color:var(--mute)}main{display:grid;grid-template-columns:minmax(0,1fr) 380px;gap:16px;padding:16px}
@media (max-width:900px){main{grid-template-columns:1fr}}
.shot{position:relative;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px;overflow:auto;max-height:82vh}
.shot svg{width:100%;height:auto;display:block}
aside{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px;align-self:start}
.cmd{font-size:20px;font-weight:600;margin:4px 0 10px}.row{margin:4px 0}.k{color:var(--mute);display:inline-block;width:92px}
.tag{display:inline-block;padding:1px 7px;border-radius:10px;border:1px solid var(--line);margin-right:4px;font-size:13px}
.tag.on{border-color:var(--warn);color:var(--warn)}ol{padding-left:20px;margin:6px 0}
.btns{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:14px}button{font:inherit;padding:9px;border-radius:6px;border:1px solid var(--line);background:var(--bg);color:var(--fg);cursor:pointer}
button.sel{outline:2px solid var(--fg)}button[data-v=correct]{color:var(--ok)}button[data-v=wrong_target],button[data-v=bad_command]{color:var(--bad)}
textarea{width:100%;min-height:60px;margin-top:10px;font:inherit;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px}
.nav{display:flex;gap:8px;margin-top:10px}.nav button{flex:1}.help{color:var(--mute);font-size:13px;margin-top:10px}
</style></head><body>
<header><b>MacBench label review</b><span id="prog"></span><span style="flex:1"></span>
<button id="export">Export JSON</button></header>
<main><div class="shot" id="shot"></div><aside id="side"></aside></main>
<script src="items.js"></script>
<script>
const KEY="__KEY__", items=window.ITEMS;
let st={i:0,v:{}};try{st=JSON.parse(localStorage.getItem(KEY))||st}catch(e){}
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){}};
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const V=[["correct","1 Correct"],["wrong_target","2 Wrong target"],["bad_command","3 Bad command / flags"],["skip","4 Skip"]];
function render(){
  const it=items[st.i], v=st.v[it.id]||{};
  const done=Object.values(st.v).filter(x=>x.verdict).length;
  document.getElementById("prog").textContent=`${st.i+1} / ${items.length} · ${done} judged`;
  const [w,h]=it.size;
  const rects=it.targets.filter(t=>t.box).map((t,j)=>`<rect x="${t.box[0]}" y="${t.box[1]}" width="${t.box[2]}" height="${t.box[3]}" fill="none" stroke="#e5322d" stroke-width="${Math.max(3,w/300)}"/><text x="${t.box[0]}" y="${Math.max(14,t.box[1]-4)}" fill="#e5322d" font-size="${Math.max(16,w/60)}" font-weight="700">${j+1}</text>`).join("");
  document.getElementById("shot").innerHTML=`<svg viewBox="0 0 ${w} ${h}"><image href="${esc(it.image)}" width="${w}" height="${h}"/>${rects}</svg>`;
  const flag=(n,on)=>`<span class="tag ${on?"on":""}">${n}: ${on?"yes":"no"}</span>`;
  const tg=it.targets.length?`<ol>${it.targets.map(t=>`<li>${esc(t.role)} "${esc(t.label)}"</li>`).join("")}</ol>`:`<div class="row"><b>None</b>: nothing on screen does this</div>`;
  document.getElementById("side").innerHTML=`
   <div class="row"><span class="k">App</span>${esc(it.app)} · ${esc(it.kind)}</div>
   <div class="cmd">“${esc(it.command)}”</div>
   <div class="row"><span class="k">Action</span>${esc(it.action)}${it.text?` · text “${esc(it.text)}” (${esc(it.text_source)})`:""}</div>
   <div class="row">${flag("absent",it.absent)}${flag("ambiguous",it.ambiguous)}${flag("irreversible",it.irreversible)}</div>
   <div class="row"><span class="k">Targets</span>(red boxes)</div>${tg}
   <div class="btns">${V.map(([k,l])=>`<button data-v="${k}" class="${v.verdict===k?"sel":""}">${l}</button>`).join("")}</div>
   <textarea id="note" placeholder="Note (optional)">${esc(v.note)}</textarea>
   <div class="nav"><button id="prev">← Prev</button><button id="next">Next →</button></div>
   <div class="help">Keys: 1–4 judge and advance · ←/→ move · n focuses the note (Esc leaves it).<br>Correct = the boxes are exactly the right element(s) and the flags/action are right.</div>`;
  document.querySelectorAll("[data-v]").forEach(b=>b.onclick=()=>judge(b.dataset.v));
  document.getElementById("note").oninput=e=>{(st.v[it.id]=st.v[it.id]||{}).note=e.target.value;save()};
  document.getElementById("prev").onclick=()=>go(-1);document.getElementById("next").onclick=()=>go(1);
}
function go(d){st.i=Math.min(items.length-1,Math.max(0,st.i+d));save();render()}
function judge(v){const it=items[st.i];(st.v[it.id]=st.v[it.id]||{}).verdict=v;st.v[it.id].kind=it.kind;st.v[it.id].app=it.app;save();go(1)}
document.addEventListener("keydown",e=>{
  if(e.target.tagName==="TEXTAREA"){if(e.key==="Escape")e.target.blur();return}
  if("1234".includes(e.key)&&e.key.length===1)judge(V[+e.key-1][0]);
  else if(e.key==="ArrowRight")go(1);else if(e.key==="ArrowLeft")go(-1);
  else if(e.key==="n"){e.preventDefault();document.getElementById("note").focus()}
});
document.getElementById("export").onclick=()=>{
  const blob=new Blob([JSON.stringify({bench:KEY,exported:new Date().toISOString(),verdicts:st.v},null,1)],{type:"application/json"});
  const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download=KEY+".json";a.click();
};
render();
</script></body></html>
"""

if __name__ == "__main__":
    main()
