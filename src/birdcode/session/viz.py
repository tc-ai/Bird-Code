# src/birdcode/session/viz.py
"""会话 jsonl → 交互式 HTML 执行流程树。

读主会话 jsonl,利用 uuid/parentUuid 树结构把每行渲染成一张卡片,自动挂载同目录下
subagents/ 的子 agent 侧链为分支,生成自包含 HTML(内嵌 CSS+JS,零依赖、零联网)。

导出:
- render_session_html(jsonl_path) -> str
- resolve_session_jsonl(sid_or_path) -> Path

通过 `birdcode session viz <sid>`(见 cli/app.py)或 scripts/visualize_session.py 调用。
视觉层纯手写 CSS(不引入框架),仅用标准库。
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path

from birdcode.utils.paths import find_project_root

TYPE_STYLE: dict[str, dict[str, str]] = {
    "user": {"color": "#3b82f6", "icon": "👤", "label": "user"},
    "assistant": {"color": "#10b981", "icon": "🤖", "label": "assistant"},
    "system": {"color": "#8b5cf6", "icon": "🚧", "label": "system"},
    "queue-operation": {"color": "#f59e0b", "icon": "📨", "label": "queue-op"},
}
UNKNOWN_STYLE = {"color": "#6b7280", "icon": "❓", "label": "unknown"}

PREVIEW_LEN = 160
ARGS_LEN = 120


@dataclass
class Block:
    kind: str
    text: str = ""
    sig: str = ""
    tool_use_id: str = ""
    name: str = ""
    input: dict | None = None
    content: str = ""
    is_error: bool = False


@dataclass
class Node:
    idx: int
    raw: dict
    type: str
    uuid: str
    parent_uuid: str | None
    logical_parent_uuid: str | None
    timestamp: str
    role: str
    blocks: list[Block]
    synthetic: bool
    is_sidechain: bool
    is_compact_summary: bool
    is_task_notif: bool
    subtype: str = ""
    compact_meta: dict = field(default_factory=dict)
    operation: str = ""
    agent_id: str = ""
    tool_use_id_ref: str = ""
    status: str = ""


def load_lines(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def parse_blocks(message: dict) -> list[Block]:
    blocks: list[Block] = []
    for b in message.get("content") or []:
        if not isinstance(b, dict):
            continue
        kind = b.get("type")
        if kind == "text":
            blocks.append(Block(kind="text", text=str(b.get("text", ""))))
        elif kind == "thinking":
            blocks.append(Block(
                kind="thinking",
                text=str(b.get("text", "")),
                sig=str(b.get("signature", "")),
            ))
        elif kind == "tool_use":
            inp = b.get("input")
            blocks.append(
                Block(
                    kind="tool_use",
                    tool_use_id=str(b.get("id", "")),
                    name=str(b.get("name", "")),
                    input=inp if isinstance(inp, dict) else {},
                )
            )
        elif kind == "tool_result":
            blocks.append(
                Block(kind="tool_result", tool_use_id=str(b.get("tool_use_id", "")),
                      content=str(b.get("content", "")), is_error=bool(b.get("is_error", False)))
            )
    return blocks


def to_node(idx: int, raw: dict) -> Node:
    t = raw.get("type", "unknown")
    message = raw.get("message") if isinstance(raw.get("message"), dict) else {}
    cmd = raw.get("compactMetadata")
    return Node(
        idx=idx, raw=raw, type=t,
        uuid=str(raw.get("uuid", "")),
        parent_uuid=raw.get("parentUuid"),
        logical_parent_uuid=raw.get("logicalParentUuid"),
        timestamp=str(raw.get("timestamp", "")),
        role=str(message.get("role", "")),
        blocks=parse_blocks(message) if t in ("user", "assistant") else [],
        synthetic=bool(raw.get("synthetic")),
        is_sidechain=bool(raw.get("isSidechain")),
        is_compact_summary=bool(raw.get("isCompactSummary")),
        is_task_notif=bool(raw.get("isTaskNotification")),
        subtype=str(raw.get("subtype", "")),
        compact_meta=cmd if isinstance(cmd, dict) else {},
        operation=str(raw.get("operation", "")),
        agent_id=str(raw.get("agentId", "")),
        tool_use_id_ref=str(raw.get("toolUseId", "")),
        status=str(raw.get("status", "")),
    )


def preview(s: str, n: int = PREVIEW_LEN) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n] + " …"


def summarize(node: Node) -> tuple[str, str]:
    t = node.type
    if t == "user":
        if node.is_compact_summary:
            m = node.compact_meta or {}
            return ("压缩摘要", f"pre={m.get('preTokens', '?')} post={m.get('postTokens', '?')} "
                    f"trigger={m.get('trigger', '?')}")
        if node.is_task_notif:
            txt = next((b.text for b in node.blocks if b.kind == "text"), "")
            return ("任务通知", preview(txt))
        tr = next((b for b in node.blocks if b.kind == "tool_result"), None)
        if tr is not None:
            return (f"工具结果 · {'ERROR' if tr.is_error else 'ok'}", preview(tr.content))
        txt = next((b.text for b in node.blocks if b.kind == "text"), "")
        return ("用户", preview(txt))
    if t == "assistant":
        parts: list[str] = []
        tu = next((b for b in node.blocks if b.kind == "tool_use"), None)
        if tu is not None:
            args = preview(json.dumps(tu.input, ensure_ascii=False), ARGS_LEN)
            parts.append(f"🔧 {tu.name}({args})")
        if any(b.kind == "thinking" for b in node.blocks):
            th = next(b for b in node.blocks if b.kind == "thinking")
            parts.append(f"💭 thinking({len(th.text)} 字)")
        txts = [b.text for b in node.blocks if b.kind == "text"]
        if txts:
            parts.append("💬 " + preview(" ".join(txts)))
        return ("助手", "  ".join(parts) if parts else "(空)")
    if t == "system":
        m = node.compact_meta or {}
        return (f"system · {node.subtype or '?'}",
                f"trigger={m.get('trigger', '?')} pre={m.get('preTokens', '?')} "
                f"post={m.get('postTokens', '?')} preserved={m.get('preservedMessages', '?')}")
    if t == "queue-operation":
        return (
            node.operation,
            f"agent={node.agent_id} status={node.status} tui={node.tool_use_id_ref}",
        )
    return ("?", str(node.raw.get("type")))


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def detect_sidechains(main_path: Path) -> dict[str, Path]:
    sc_dir = main_path.parent / main_path.stem / "subagents"
    out: dict[str, Path] = {}
    if not sc_dir.is_dir():
        return out
    for p in sorted(sc_dir.glob("agent-*.jsonl")):
        agent_id = ""
        for raw in load_lines(p):
            aid = raw.get("agentId")
            if isinstance(aid, str) and aid:
                agent_id = aid
                break
        out[agent_id or p.stem] = p
    return out


def build_spawn_map(main_nodes: list[Node]) -> dict[str, str]:
    spawn: dict[str, str] = {}
    for n in main_nodes:
        tur = n.raw.get("toolUseResult")
        if isinstance(tur, dict):
            for tuid, obj in tur.items():
                if isinstance(obj, dict) and obj.get("agentId"):
                    spawn.setdefault(str(obj["agentId"]), str(tuid))
        if n.type == "queue-operation" and n.agent_id and n.tool_use_id_ref:
            spawn.setdefault(n.agent_id, n.tool_use_id_ref)
    return spawn


def tuid_to_main_idx(main_nodes: list[Node]) -> dict[str, int]:
    out: dict[str, int] = {}
    for n in main_nodes:
        for b in n.blocks:
            if b.kind == "tool_use" and b.tool_use_id:
                out[b.tool_use_id] = n.idx
    return out


def ts_short(ts: str) -> str:
    return ts[11:19] if len(ts) >= 19 else ts


def render_block_full(b: Block) -> str:
    if b.kind == "text":
        return f"<p class='blk'><b>text:</b> {esc(b.text)}</p>"
    if b.kind == "thinking":
        return f"<p class='blk'><b>thinking:</b> {esc(b.text)}</p>"
    if b.kind == "tool_use":
        args = esc(json.dumps(b.input, ensure_ascii=False))
        return (
            f"<p class='blk'><b>tool_use:</b> {esc(b.name)}({args})"
            f" <code>id={esc(b.tool_use_id)}</code></p>"
        )
    if b.kind == "tool_result":
        tag = " ERROR" if b.is_error else ""
        return (
            f"<p class='blk'><b>tool_result{tag}:</b> {esc(b.content)}"
            f" <code>↤ {esc(b.tool_use_id)}</code></p>"
        )
    return ""


def node_card(node: Node, prefix: str = "m") -> str:
    style = TYPE_STYLE.get(node.type, UNKNOWN_STYLE)
    title, body = summarize(node)
    flags = []
    if node.is_sidechain:
        flags.append("侧链")
    if node.synthetic:
        flags.append("合成")
    flag_html = "".join(f'<span class="flag">{esc(f)}</span>' for f in flags)
    extra_cls = ""
    if node.type == "user" and any(b.kind == "tool_result" for b in node.blocks):
        extra_cls = " is-result"
    if node.type == "system":
        extra_cls = " is-system"
    raw_pretty = json.dumps(node.raw, ensure_ascii=False, indent=2)
    if node.blocks:
        full = "".join(render_block_full(b) for b in node.blocks)
    else:
        full = "<em>(无 content)</em>"
    icon = style["icon"]
    label = esc(style["label"])
    color = style["color"]
    return f"""
    <article class="card{extra_cls}" id="{prefix}{node.idx}"
             data-type="{esc(node.type)}" style="--c:{color}">
      <div class="card-main">
        <header class="card-head">
          <span class="chip"><span class="chip-ico">{icon}</span>{label}</span>
          <span class="seq">#{node.idx}</span>
          <span class="title">{esc(title)}</span>
          {flag_html}
          <span class="ts">{esc(ts_short(node.timestamp))}</span>
        </header>
        <div class="preview">{esc(body)}</div>
        <details class="detail"><summary>展开详情 / 原始 JSON</summary>
          <div class="full">{full}</div>
          <pre class="raw">{esc(raw_pretty)}</pre>
        </details>
      </div>
    </article>"""


def render_sidechain(path: Path, agent_id: str) -> str:
    nodes = [to_node(i, r) for i, r in enumerate(load_lines(path))]
    cards = "\n".join(node_card(n, prefix="s") for n in nodes)
    return f"""
    <div class="sidechain">
      <div class="sc-head" onclick="this.parentElement.classList.toggle('collapsed')">
        <span class="sc-tri">▼</span> 🔀 子 agent <code>{esc(agent_id)}</code>
        <span class="sc-meta">{len(nodes)} 步 · {esc(path.name)}</span>
      </div>
      <div class="sc-body">{cards}</div>
    </div>"""


CSS = """
:root{
  --bg-page:#f4f5f7; --bg-card:#ffffff; --bg-soft:#f7f8fa;
  --text:#1f2330; --text-dim:#5b6477; --text-faint:#9aa3b2;
  --border:#e6e8ec; --spine:#cbd0d8;
  --shadow:0 1px 2px rgba(16,24,40,.05), 0 1px 3px rgba(16,24,40,.04);
  --shadow-h:0 10px 24px rgba(16,24,40,.12), 0 2px 6px rgba(16,24,40,.06);
}
[data-theme="dark"]{
  --bg-page:#0b1220; --bg-card:#141d2f; --bg-soft:#0f1729;
  --text:#e6edf6; --text-dim:#9aa7bd; --text-faint:#5e6b82;
  --border:#243049; --spine:#2a3a55;
  --shadow:0 1px 2px rgba(0,0,0,.4);
  --shadow-h:0 10px 28px rgba(0,0,0,.55);
}
*{box-sizing:border-box;}
body{margin:0;background:var(--bg-page);color:var(--text);
  font-family:-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",
  Roboto,sans-serif;line-height:1.55;}
header{position:sticky;top:0;z-index:30;background:linear-gradient(135deg,#0f172a,#1e293b);
  color:#e5e7eb;padding:11px 18px;box-shadow:0 2px 14px rgba(0,0,0,.28);}
header .row{display:flex;align-items:center;gap:10px;flex-wrap:wrap;}
header h1{margin:0;font-size:15px;font-weight:650;}
.ver{font-size:11px;color:#fcd34d;background:rgba(245,158,11,.16);
  border:1px solid rgba(245,158,11,.4);padding:2px 8px;border-radius:10px;}
.spacer{flex:1 1 auto;}
#search{background:#1f2937;color:#e5e7eb;border:1px solid #475569;border-radius:8px;
  padding:5px 10px;font-size:12px;width:180px;}
#search::placeholder{color:#6b7280;}
#search:focus{outline:none;border-color:#3b82f6;}
#search-count{font-size:11px;color:#9ca3af;min-width:34px;text-align:center;}
.zoom-ctrl{display:inline-flex;align-items:center;gap:1px;background:rgba(255,255,255,.06);
  border:1px solid rgba(255,255,255,.1);border-radius:8px;padding:2px;}
.zoom-ctrl button{background:transparent;color:#e5e7eb;border:none;width:26px;height:24px;
  border-radius:6px;cursor:pointer;font-size:13px;}
.zoom-ctrl button:hover{background:rgba(255,255,255,.14);}
#zoom-level{color:#9ca3af;font-size:11px;min-width:34px;text-align:center;}
.tog{background:#334155;color:#e5e7eb;border:1px solid #475569;border-radius:8px;
  padding:5px 10px;font-size:12px;cursor:pointer;}
.tog:hover{background:#475569;}
.meta{font-size:12px;color:#9ca3af;word-break:break-all;margin-top:6px;}
.meta b{color:#cbd5e1;font-weight:600;}
.legend{display:flex;gap:7px;flex-wrap:wrap;margin-top:9px;}
.lg{display:inline-flex;align-items:center;gap:6px;cursor:pointer;font-size:12px;
  background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);
  padding:3px 10px;border-radius:14px;transition:.15s;user-select:none;color:#e5e7eb;}
.lg:hover{background:rgba(255,255,255,.12);}
.lg.off{opacity:.32;text-decoration:line-through;}
.lg .d{width:9px;height:9px;border-radius:3px;}
.lg b{color:#fff;}
main{max-width:1080px;margin:0 auto;padding:26px 34px 80px;}
.trunk{position:relative;padding-left:28px;}
.trunk::before{content:"";position:absolute;left:9px;top:10px;bottom:10px;width:2px;
  background:var(--spine);border-radius:2px;}
.card{position:relative;background:var(--bg-card);border:1px solid var(--border);
  border-left:4px solid var(--c);border-radius:11px;box-shadow:var(--shadow);
  margin-bottom:14px;transition:box-shadow .15s,transform .15s;}
.card:hover{box-shadow:var(--shadow-h);transform:translateY(-1px);}
.card::before{content:"";position:absolute;left:-23px;top:17px;width:12px;height:12px;
  border-radius:50%;background:var(--c);box-shadow:0 0 0 3px var(--bg-page);}
.card.hidden{display:none;}
.card.hit{box-shadow:0 0 0 3px #f59e0b,var(--shadow-h);border-color:#f59e0b;}
.card.dim{opacity:.2;}
.card-main{padding:10px 14px;}
.card-head{display:flex;align-items:center;gap:8px;flex-wrap:wrap;font-size:12px;}
.chip{display:inline-flex;align-items:center;gap:5px;background:var(--c);color:#fff;
  padding:2px 10px;border-radius:11px;font-weight:600;font-size:11px;white-space:nowrap;}
.chip-ico{font-size:12px;}
.seq{color:var(--text-faint);font-variant-numeric:tabular-nums;font-size:11px;}
.title{font-weight:650;color:var(--text);}
.flag{font-size:10px;background:#fee2e2;color:#b91c1c;
  padding:1px 7px;border-radius:8px;font-weight:600;}
[data-theme="dark"] .flag{background:#7f1d1d;color:#fecaca;}
.ts{margin-left:auto;color:var(--text-faint);font-size:11px;font-variant-numeric:tabular-nums;}
.preview{margin:6px 0 0;font-size:13.5px;color:var(--text-dim);word-break:break-word;}
.card[data-type="assistant"] .preview{color:var(--text);}
.card.is-system{background:color-mix(in srgb,var(--c) 12%,var(--bg-card));}
.detail{margin-top:8px;}
.detail summary{cursor:pointer;font-size:11px;color:var(--text-faint);}
.detail[open] summary{color:var(--text-dim);margin-bottom:4px;}
.full{margin-top:6px;font-size:12.5px;}
.full .blk{margin:3px 0;padding:6px 9px;background:var(--bg-soft);border-radius:6px;
  border:1px solid var(--border);word-break:break-word;}
.full code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:11px;color:var(--text-faint);}
.raw{margin-top:8px;background:#0f172a;color:#e2e8f0;padding:10px;border-radius:8px;font-size:11.5px;
  overflow:auto;max-height:380px;white-space:pre-wrap;word-break:break-word;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;border:1px solid #1e293b;}
.sidechain{margin:2px 0 14px 22px;border-left:2px dashed #f59e0b;padding-left:14px;}
.sc-head{cursor:pointer;font-size:12.5px;color:#92400e;background:#fffbeb;border:1px solid #fde68a;
  padding:6px 11px;border-radius:8px;user-select:none;display:flex;align-items:center;gap:8px;}
[data-theme="dark"] .sc-head{color:#fcd34d;background:rgba(245,158,11,.12);
  border-color:rgba(245,158,11,.35);}
.sc-tri{display:inline-block;transition:transform .15s;}
.sidechain.collapsed .sc-tri{transform:rotate(-90deg);}
.sidechain.collapsed .sc-body{display:none;}
.sc-meta{color:var(--text-faint);font-size:11px;margin-left:auto;}
.sc-meta code,.sc-head code{font-family:ui-monospace,SFMono-Regular,Menlo,
  Consolas,monospace;font-size:11px;}
.sc-body{margin-top:10px;}
.sc-group{margin:6px 0 14px 22px;border-left:2px dotted var(--spine);padding-left:14px;}
.sc-group-head{font-size:12px;color:var(--text-dim);margin-bottom:8px;}
#minimap{position:fixed;right:8px;top:64px;bottom:14px;width:16px;
  z-index:20;display:flex;
  flex-direction:column;gap:2px;padding:5px 3px;background:var(--bg-card);
  border:1px solid var(--border);
  border-radius:8px;opacity:.9;overflow:hidden;}
.mm-dot{width:9px;height:7px;border-radius:2px;cursor:pointer;opacity:.65;
  flex:0 0 auto;align-self:center;transition:.1s;}
.mm-dot:hover{opacity:1;transform:scale(1.4);}
.mm-dot.cur{opacity:1;box-shadow:0 0 0 2px var(--text);}
body.dragging, body.dragging *{cursor:grabbing!important;}
footer{text-align:center;color:var(--text-faint);font-size:11px;padding:24px;line-height:1.8;}
@media(max-width:820px){ #minimap{display:none;} main{padding-right:20px;} #search{width:130px;} }
"""

JS = """
(function(){
  /* theme */
  try{document.documentElement.setAttribute('data-theme',localStorage.getItem('bc-theme')||'light');}catch(e){
    document.documentElement.setAttribute('data-theme','light');}
  var tog=document.querySelector('.tog');
  if(tog) tog.addEventListener('click',function(){
    var cur=document.documentElement.getAttribute('data-theme');
    var n=cur==='dark'?'light':'dark';
    document.documentElement.setAttribute('data-theme',n);
    try{localStorage.setItem('bc-theme',n);}catch(e){}
  });

  /* type filter */
  document.querySelectorAll('.legend .lg').forEach(function(el){
    el.addEventListener('click',function(){
      el.classList.toggle('off');
      var off={};
      document.querySelectorAll('.legend .lg.off').forEach(function(e){off[e.dataset.t]=1;});
      document.querySelectorAll('.card').forEach(function(c){
        c.classList.toggle('hidden',!!off[c.dataset.type]);});
      rebuildMinimap(); markCurrent();
    });
  });

  /* zoom(CSS zoom → 自动 reflow + 页面滚动即可平移) */
  var main=document.querySelector('main');
  var z=1,ZMIN=0.5,ZMAX=2.0;
  var zLabel=document.getElementById('zoom-level');
  function setZoom(v){z=Math.max(ZMIN,Math.min(ZMAX,Math.round(v*100)/100));main.style.zoom=z;
    if(zLabel)zLabel.textContent=Math.round(z*100)+'%';}
  setZoom(1);
  document.getElementById('zoom-in').addEventListener('click',function(){setZoom(z+0.1);});
  document.getElementById('zoom-out').addEventListener('click',function(){setZoom(z-0.1);});
  document.getElementById('zoom-reset').addEventListener('click',function(){setZoom(1);});
  window.addEventListener('wheel',function(e){if(e.ctrlKey){e.preventDefault();setZoom(z+(e.deltaY<0?0.1:-0.1));}},{passive:false});

  /* drag-to-pan(拖拽滚动页面) */
  var dragging=false,sx=0,sy=0;
  function downable(t){return !t.closest('.card,header,#minimap,input,button,details,.lg');}
  document.addEventListener('mousedown',function(e){
    if(e.button!==0||!downable(e.target))return;
    dragging=true;sx=e.clientX;sy=e.clientY;document.body.classList.add('dragging');
  });
  window.addEventListener('mousemove',function(e){if(!dragging)return;
    window.scrollBy(sx-e.clientX,sy-e.clientY);sx=e.clientX;sy=e.clientY;});
  window.addEventListener('mouseup',function(){dragging=false;document.body.classList.remove('dragging');});

  /* search */
  var sb=document.getElementById('search'),sc=document.getElementById('search-count');
  var hits=[],hi=-1;
  function clearH(){document.querySelectorAll('.card.hit,.card.dim').forEach(function(c){
    c.classList.remove('hit');c.classList.remove('dim');});}
  function runSearch(){
    var q=(sb.value||'').trim().toLowerCase();clearH();hits=[];hi=-1;
    if(!q){sc.textContent='';return;}
    document.querySelectorAll('.card:not(.hidden)').forEach(function(c){
      if(c.textContent.toLowerCase().indexOf(q)>=0){c.classList.add('hit');hits.push(c);}
      else{c.classList.add('dim');}});
    if(hits.length){hi=0;sc.textContent='1/'+hits.length;
      hits[0].scrollIntoView({block:'center',behavior:'smooth'});}
    else sc.textContent='0/0';
  }
  sb.addEventListener('input',runSearch);
  sb.addEventListener('keydown',function(e){
    if(!hits.length)return;
    if(e.key==='Enter'){e.preventDefault();hi=(hi+1)%hits.length;sc.textContent=(hi+1)+'/'+hits.length;
      hits[hi].scrollIntoView({block:'center',behavior:'smooth'});}
    if(e.key==='Escape'){sb.value='';runSearch();sb.blur();}
  });

  /* minimap */
  var rail=document.getElementById('minimap');
  var COL={user:'#3b82f6',assistant:'#10b981',system:'#8b5cf6','queue-operation':'#f59e0b'};
  function rebuildMinimap(){
    rail.innerHTML='';
    document.querySelectorAll('.card:not(.hidden)').forEach(function(c){
      var d=document.createElement('div');d.className='mm-dot';d.dataset.id=c.id;
      d.style.background=COL[c.dataset.type]||'#888';
      var t=c.querySelector('.title');d.title='#'+c.id+' '+(t?t.textContent:'');
      d.addEventListener('click',function(){c.scrollIntoView({block:'start',behavior:'smooth'});});
      rail.appendChild(d);
    });
  }
  rebuildMinimap();
  var raf=0;
  function markCurrent(){raf=0;var top=130,best=null,bestD=Infinity;
    document.querySelectorAll('.card:not(.hidden)').forEach(function(c){
      var r=c.getBoundingClientRect();if(r.bottom<0)return;
      var d=Math.abs(r.top-top);
      if(d<bestD){bestD=d;best=c;}});
    document.querySelectorAll('.mm-dot.cur').forEach(function(d){d.classList.remove('cur');});
    if(best){var dot=rail.querySelector('.mm-dot[data-id="'+best.id+'"]');
      if(dot)dot.classList.add('cur');}}
  window.addEventListener('scroll',function(){if(!raf)raf=requestAnimationFrame(markCurrent);});
  markCurrent();
})();
"""


def render_session_html(main_path: Path) -> str:
    main_nodes = [to_node(i, r) for i, r in enumerate(load_lines(main_path))]
    sidechains = detect_sidechains(main_path)
    spawn = build_spawn_map(main_nodes)
    tuid_idx = tuid_to_main_idx(main_nodes)
    agentid_main_idx: dict[str, int] = {}
    for n in main_nodes:
        if n.agent_id and n.agent_id not in agentid_main_idx:
            agentid_main_idx[n.agent_id] = n.idx
    attach: dict[int, list[str]] = {}
    standalone: list[str] = []
    for aid in sidechains:
        tuid = spawn.get(aid)
        mi = tuid_idx.get(tuid) if tuid else None
        if mi is None:
            mi = agentid_main_idx.get(aid)
        if mi is not None:
            attach.setdefault(mi, []).append(aid)
        else:
            standalone.append(aid)

    counts: dict[str, int] = {}
    for n in main_nodes:
        counts[n.type] = counts.get(n.type, 0) + 1

    parts: list[str] = []
    for n in main_nodes:
        parts.append(node_card(n, prefix="m"))
        for aid in attach.get(n.idx, []):
            parts.append(render_sidechain(sidechains[aid], aid))
    if standalone:
        parts.append('<div class="sc-group"><div class="sc-group-head">🔀 未关联到主流程的子 agent '
                     f'({len(standalone)})</div>'
                     + "\n".join(
                         render_sidechain(sidechains[a], a) for a in standalone
                     )
                     + "</div>")

    legend_items = "".join(
        f'<span class="lg" data-t="{esc(t)}">'
        f'<span class="d" style="background:{s["color"]}"></span>'
        f'{s["icon"]} {s["label"]} <b>{counts.get(t, 0)}</b></span>'
        for t, s in TYPE_STYLE.items()
    )
    meta = (f"<b>文件:</b> {esc(str(main_path))} &nbsp;·&nbsp; "
            f"<b>主流程:</b> {len(main_nodes)} 行 &nbsp;·&nbsp; "
            f"<b>子 agent:</b> {len(sidechains)} &nbsp;·&nbsp; "
            f"{' '.join(f'{k}:{v}' for k, v in counts.items())}")
    trunk = "\n".join(parts)
    title = f"BirdCode 会话: {main_path.stem}"
    return (
        "<!doctype html>\n<html lang=\"zh\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>{esc(title)}</title>\n"
        "<script>try{document.documentElement.setAttribute('data-theme',"
        "localStorage.getItem('bc-theme')||'light');}catch(e){"
        "document.documentElement.setAttribute('data-theme','light');}</script>\n"
        f"<style>{CSS}</style></head>\n<body>\n"
        "<header><div class=\"row\"><h1>🐦 BirdCode 会话执行流程树</h1>"
        "<span class=\"ver\">v3 · 🔍搜索 ＋ 缩放 ＋ 迷你地图</span>"
        "<input id=\"search\" type=\"search\" placeholder=\"🔍 搜索节点内容…\" />"
        "<span id=\"search-count\"></span>"
        "<span class=\"spacer\"></span>"
        "<span class=\"zoom-ctrl\">"
        "<button id=\"zoom-out\" title=\"缩小\">－</button>"
        "<span id=\"zoom-level\">100%</span>"
        "<button id=\"zoom-in\" title=\"放大\">＋</button>"
        "<button id=\"zoom-reset\" title=\"重置\">⟳</button></span>"
        "<button class=\"tog\">🌙</button></div>"
        f"<div class=\"meta\">{meta}</div>"
        f"<div class=\"legend\">{legend_items}</div></header>\n"
        f"<main><div class=\"trunk\">\n{trunk}\n</div></main>\n"
        "<aside id=\"minimap\"></aside>\n"
        "<footer>每节点 = 一行 jsonl · 左侧时间线为主流程 · 橙色虚线分支为子 agent 侧链 · "
        "顶栏可搜索/缩放(或 Ctrl+滚轮) · 空白处可拖拽平移 · "
        "右侧迷你地图点击跳转 · 点节点「展开」看原始 JSON</footer>\n"
        f"<script>{JS}</script>\n</body></html>"
    )


def resolve_session_jsonl(sid: str) -> Path:
    """sid 是已有文件路径 → 直接用;否则当 sessionId 在默认项目会话根下找。

    找不到 → FileNotFoundError。
    """
    from birdcode.session import paths as session_paths

    p = Path(sid)
    if p.is_file():
        return p.resolve()
    project_root = find_project_root(Path.cwd())
    jf = session_paths.session_jsonl(session_paths.default_root(), sid, project_root)
    if jf.is_file():
        return jf
    raise FileNotFoundError(
        f"找不到会话: {sid!r}(可传 jsonl 路径或 sessionId;TUI 内 /sessions 查看可用 id)。"
    )
