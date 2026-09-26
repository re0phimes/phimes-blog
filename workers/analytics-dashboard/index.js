/**
 * analytics-dashboard —— 只读的站点统计看板
 *
 * 挂在 stats.phimes.top/*（比 site-analytics 的 *.phimes.top/* 更具体，优先级更高）
 *
 * 数据来源：Cloudflare Analytics Engine
 *   site_visits —— 真实读者（blob9 = 'browser' | 'feed'）
 *   bot_visits  —— 爬虫 / 扫描器 / 空 UA / 404 探测（blob8 = kind）
 *
 * blob 布局（v2 起）
 *   site_visits: 1 path | 2 ipHash | 3 ""(原明文IP) | 4 referer | 5 country
 *                6 city | 7 hostname | 8 ua | 9 clientClass | 10 status
 *   bot_visits:  1 path | 2 ""(原明文IP) | 3 ua | 4 hostname | 5 country
 *                6 ipHash | 7 botType | 8 kind | 9 status
 *
 * 鉴权：HTTP Basic Auth，密码来自 secret DASH_PASSWORD
 * AE 只读 token 来自 secret CF_ANALYTICS_TOKEN，不会暴露到浏览器
 */

// 时间范围白名单 —— 只允许这些值，避免拼接 SQL
const RANGES = {
  "1h": { label: "最近 1 小时", interval: "INTERVAL '1' HOUR", bucket: "hour" },
  "6h": { label: "最近 6 小时", interval: "INTERVAL '6' HOUR", bucket: "hour" },
  "24h": { label: "最近 24 小时", interval: "INTERVAL '24' HOUR", bucket: "hour" },
  "7d": { label: "最近 7 天", interval: "INTERVAL '7' DAY", bucket: "day" },
  "30d": { label: "最近 30 天", interval: "INTERVAL '30' DAY", bucket: "day" },
  "90d": { label: "最近 90 天", interval: "INTERVAL '90' DAY", bucket: "day" },
};
const DEFAULT_RANGE = "7d";

function unauthorized() {
  return new Response("需要登录", {
    status: 401,
    headers: { "WWW-Authenticate": "Basic realm=\"analytics\", charset=\"UTF-8\"" },
  });
}

function checkAuth(request, env) {
  const expected = env.DASH_PASSWORD || "";
  if (!expected) return false;
  const header = request.headers.get("authorization") || "";
  if (!header.startsWith("Basic ")) return false;
  try {
    const decoded = atob(header.slice(6));
    const idx = decoded.indexOf(":");
    return (idx >= 0 ? decoded.slice(idx + 1) : decoded) === expected;
  } catch {
    return false;
  }
}

async function aeQuery(env, sql) {
  const res = await fetch(
    `https://api.cloudflare.com/client/v4/accounts/${env.CF_ACCOUNT_ID}/analytics_engine/sql`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${env.CF_ANALYTICS_TOKEN}`,
        "Content-Type": "text/plain",
      },
      body: sql,
    },
  );
  const text = await res.text();
  if (!res.ok) throw new Error(`AE ${res.status}: ${text.slice(0, 200)}`);
  return JSON.parse(text);
}

// 设备归类表达式（Analytics Engine 不支持 multiIf / CASE WHEN，只能嵌套 if()）
const DEVICE_EXPR =
  "if(blob8 LIKE '%iPhone%','iPhone', if(blob8 LIKE '%iPad%','iPad', if(blob8 LIKE '%Android%','Android', if(blob8 LIKE '%Macintosh%','macOS', if(blob8 LIKE '%Windows%','Windows', if(blob8 LIKE '%Linux%','Linux', '其他'))))))";

// 可下钻的维度白名单。val 由前端传入，必须经过这些定义才能拼进 SQL。
const DETAIL_DIMS = {
  bot: {
    botType: { col: "blob7", label: "爬虫类型" },
    kind: { col: "blob8", label: "分类" },
    path: { col: "blob1", label: "路径" },
    country: { col: "blob5", label: "国家" },
    asOrg: { col: "blob10", label: "AS 组织" },
  },
  human: {
    path: { col: "blob1", label: "路径" },
    country: { col: "blob5", label: "国家" },
    city: { col: "blob6", label: "城市" },
    referer: { col: "blob4", label: "来源" },
    asOrg: { col: "blob11", label: "AS 组织" },
    device: { expr: DEVICE_EXPR, label: "设备" },
  },
};

// 把值安全地放进 SQL 字面量：AE 的 SQL API 不支持参数化查询
const sqlStr = (v) =>
  "'" +
  String(v == null ? "" : v)
    .replace(/\\/g, "")
    .replace(/'/g, "''") +
  "'";

async function detail(env, rangeKey, src, dimName, val, page, pageSize) {
  const r = RANGES[rangeKey] || RANGES[DEFAULT_RANGE];
  const dimTable = DETAIL_DIMS[src];
  if (!dimTable) throw new Error("bad src");
  const dim = dimTable[dimName];
  if (!dim) throw new Error("bad dim");
  const dimSql = dim.col || dim.expr;

  const ps = Math.min(Math.max(Number(pageSize) || 50, 10), 200);
  const pg = Math.max(Number(page) || 1, 1);
  const offset = (pg - 1) * ps;
  const where = `timestamp > NOW() - ${r.interval} AND ${dimSql} = ${sqlStr(val)}`;

  const [countRes, dataRes] = await Promise.all([
    aeQuery(
      env,
      `SELECT count() AS total FROM ${src === "bot" ? "bot_visits" : "site_visits"} WHERE ${where}`,
    ),
    src === "bot"
      ? aeQuery(
          env,
          `SELECT timestamp, blob1 AS path, blob3 AS ua, blob5 AS country, blob7 AS botType, blob8 AS kind, blob9 AS status, blob10 AS asOrg, blob2 AS oldIp, blob6 AS ipHash FROM bot_visits WHERE ${where} ORDER BY timestamp DESC LIMIT ${ps} OFFSET ${offset}`,
        )
      : aeQuery(
          env,
          `SELECT timestamp, blob1 AS path, blob8 AS ua, blob5 AS country, blob6 AS city, blob4 AS referer, blob9 AS clientClass, blob10 AS status, blob11 AS asOrg, blob2 AS ipHash, blob3 AS oldIp FROM site_visits WHERE ${where} ORDER BY timestamp DESC LIMIT ${ps} OFFSET ${offset}`,
        ),
  ]);

  const total = Number((countRes.data && countRes.data[0] && countRes.data[0].total) || 0);
  return {
    src,
    dim: dimName,
    val,
    page: pg,
    pageSize: ps,
    total,
    totalPages: Math.max(1, Math.ceil(total / ps)),
    data: dataRes.data || [],
  };
}
async function collect(env, rangeKey) {
  const r = RANGES[rangeKey] || RANGES[DEFAULT_RANGE];
  const W = `timestamp > NOW() - ${r.interval}`;
  // 真人：浏览器页面浏览。旧数据没有 blob9，用 != 'feed' 兼容
  const HUMAN = `${W} AND blob9 != 'feed'`;
  const FEED = `${W} AND blob9 = 'feed'`;
  const BOTS = W;
  const tExpr = r.bucket === "hour" ? "toStartOfHour(timestamp)" : "toDate(timestamp)";

  const q = (sql) => aeQuery(env, sql);

  const [
    human,
    feed,
    bots,
    trendHuman,
    trendBot,
    kinds,
    pages,
    countries,
    referers,
    devices,
    postHuman,
    postBot,
  ] = await Promise.all([
    q(`SELECT count() AS pv, count(DISTINCT blob2) AS uv FROM site_visits WHERE ${HUMAN}`),
    q(`SELECT count() AS pv, count(DISTINCT blob2) AS uv FROM site_visits WHERE ${FEED}`),
    q(`SELECT count() AS n, count(DISTINCT blob6) AS uniq FROM bot_visits WHERE ${BOTS}`),
    q(
      `SELECT ${tExpr} AS t, count() AS n FROM site_visits WHERE ${HUMAN} GROUP BY t ORDER BY t ASC`,
    ),
    q(`SELECT ${tExpr} AS t, count() AS n FROM bot_visits WHERE ${BOTS} GROUP BY t ORDER BY t ASC`),
    q(
      `SELECT blob8 AS kind, blob7 AS botType, count() AS n FROM bot_visits WHERE ${BOTS} GROUP BY kind, botType ORDER BY n DESC LIMIT 20`,
    ),
    q(
      `SELECT blob1 AS path, count() AS n FROM site_visits WHERE ${HUMAN} GROUP BY path ORDER BY n DESC LIMIT 15`,
    ),
    q(
      `SELECT blob5 AS country, count() AS n FROM site_visits WHERE ${HUMAN} GROUP BY country ORDER BY n DESC LIMIT 10`,
    ),
    q(
      `SELECT blob4 AS referer, count() AS n FROM site_visits WHERE ${HUMAN} GROUP BY referer ORDER BY n DESC LIMIT 12`,
    ),
    q(
      `SELECT ${DEVICE_EXPR} AS device, count() AS n FROM site_visits WHERE ${HUMAN} GROUP BY device ORDER BY n DESC LIMIT 8`,
    ),
    // 内容优化用：每篇文章的真人阅读量
    q(
      `SELECT blob1 AS path, count() AS n FROM site_visits WHERE ${HUMAN} AND blob1 LIKE '/posts/%' GROUP BY path ORDER BY n DESC LIMIT 60`,
    ),
    // 内容优化用：每篇文章被爬虫抓取的次数
    q(
      `SELECT blob1 AS path, count() AS n FROM bot_visits WHERE ${BOTS} AND blob1 LIKE '/posts/%' GROUP BY path ORDER BY n DESC LIMIT 60`,
    ),
  ]);

  const num = (res, k) => Number((res.data && res.data[0] && res.data[0][k]) || 0);
  return {
    range: rangeKey in RANGES ? rangeKey : DEFAULT_RANGE,
    rangeLabel: r.label,
    bucket: r.bucket,
    human: { pv: num(human, "pv"), uv: num(human, "uv") },
    feed: { pv: num(feed, "pv"), uv: num(feed, "uv") },
    bots: { pv: num(bots, "n"), uv: num(bots, "uniq") },
    trendHuman: trendHuman.data || [],
    trendBot: trendBot.data || [],
    kinds: kinds.data || [],
    pages: pages.data || [],
    countries: countries.data || [],
    referers: referers.data || [],
    devices: devices.data || [],
    postHuman: postHuman.data || [],
    postBot: postBot.data || [],
  };
}

const PAGE = `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<link rel="icon" type="image/png" href="https://blog.phimes.top/images/logo/favicon-32x32.png">
<link rel="apple-touch-icon" href="https://blog.phimes.top/images/logo/favicon-192x192.png">
<title>站点统计 · Phimes</title>
<style>
  :root{--bg:#faf9f5;--card:#fff;--line:#e6dfd8;--ink:#141413;--sub:#6b6a64;--coral:#cc785c;--teal:#2f8574;--amber:#b8791f;--blue:#6a9bcc}
  *{box-sizing:border-box}
  body{margin:0;padding:20px;background:var(--bg);color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
    font-size:14px;line-height:1.6}
  .top{display:flex;align-items:center;gap:11px;margin-bottom:16px}
  .top .brand{width:36px;height:36px;display:block;flex:none}
  .top .brand-text{min-width:0}
  h1{font-size:20px;margin:0;line-height:1.3}
  .meta{color:var(--sub);font-size:12px;margin:1px 0 0}
  .ranges{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:18px}
  .ranges button{border:1px solid var(--line);background:var(--card);color:var(--ink);
    border-radius:8px;padding:5px 12px;font-size:13px;cursor:pointer;font-family:inherit;transition:all .2s}
  .ranges button:hover{border-color:var(--coral);color:var(--coral)}
  .ranges button.on{background:var(--coral);border-color:var(--coral);color:#fff}
  .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:18px}
  .kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:13px 15px}
  .kpi .k{color:var(--sub);font-size:12px}
  .kpi .v{font-size:25px;font-weight:600;letter-spacing:-.02em;margin-top:2px}
  .kpi .s{color:var(--sub);font-size:11px;margin-top:2px}
  .coral .v{color:var(--coral)} .teal .v{color:var(--teal)} .amber .v{color:var(--amber)} .blue .v{color:var(--blue)}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:15px}
  .card h2{font-size:13px;margin:0 0 10px;color:var(--sub);font-weight:600}
  table{width:100%;border-collapse:collapse;font-size:13px}
  td,th{padding:4px 0;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}
  th{color:var(--sub);font-weight:500;font-size:11px}
  td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
  tr:last-child td{border-bottom:0}
  .bar{display:inline-block;height:6px;border-radius:3px;vertical-align:middle;margin-left:8px;opacity:.5;background:var(--coral)}
  .bar.teal{background:var(--teal)} .bar.amber{background:var(--amber)} .bar.blue{background:var(--blue)}
  .chart{display:flex;align-items:flex-end;gap:2px;height:110px;margin-top:6px}
  .chart i{flex:1;border-radius:2px 2px 0 0;min-height:2px;background:var(--coral);opacity:.8}
  .chart i.bot{background:var(--sub);opacity:.35}
  .legend{color:var(--sub);font-size:11px;margin-top:8px;display:flex;gap:14px;flex-wrap:wrap}
  .dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px;vertical-align:middle}
  .note{color:var(--sub);font-size:11px;margin-top:18px;line-height:1.8}
  code{background:var(--card);border:1px solid var(--line);border-radius:4px;padding:1px 5px;font-size:11px}
  .err{background:#fff5f5;border:1px solid #e6c9c9;color:#a33;padding:12px;border-radius:8px}
  .muted{color:var(--sub);font-size:11px}
  .tag{display:inline-block;font-size:10px;padding:1px 6px;border-radius:999px;border:1px solid var(--line);color:var(--sub)}
  .tag.good{color:var(--teal);border-color:#bcd9d0}
  .tag.warn{color:var(--amber);border-color:#e6d5b0}
  .tag.bad{color:#a33;border-color:#e6c9c9}
  .full{grid-column:1/-1}
  #detail{overflow-x:auto}
  #detail table{min-width:900px}
  tr.rowlink{cursor:pointer}
  tr.rowlink:hover td{background:rgba(204,120,92,.07)}
  tr.rowlink td:first-child::before{content:"› ";color:var(--coral);opacity:.6}
  .pager{display:flex;flex-wrap:wrap;align-items:center;gap:10px;justify-content:space-between;margin:10px 0}
  .pg-group{display:flex;align-items:center;gap:4px;flex-wrap:wrap}
  button.pg{border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:8px;
    padding:3px 9px;font-size:12px;cursor:pointer;font-family:inherit;transition:all .18s}
  button.pg:hover:not(:disabled){border-color:var(--coral);color:var(--coral)}
  button.pg.on{background:var(--coral);border-color:var(--coral);color:#fff}
  button.pg:disabled{opacity:.35;cursor:default}
</style></head><body>
<div class="top">
  <img class="brand" src="https://blog.phimes.top/images/logo/logo.webp" alt="Phimes" width="36" height="36">
  <div class="brand-text">
    <h1>站点统计</h1>
    <div class="meta" id="meta">加载中…</div>
  </div>
</div>
<div class="ranges" id="ranges"></div>
<div id="app"></div>
<div id="detail" class="card" style="display:none;margin-top:16px;overflow-x:auto"></div>
<div class="note">
  <b>口径说明</b><br>
  · <b>真人</b> = 返回 200 的页面浏览（静态资源不计）。RSS 阅读器单独记为「订阅者」。<br>
  · <b>爬虫/扫描</b> = 爬虫 UA、扫描器、空 UA、以及所有 404 探测。<br>
  · 判定用<strong>正向白名单</strong>：路径必须属于 <code>/</code>、<code>/posts/</code>、<code>/pages/</code>、<code>/page</code>
  &nbsp;以及 <code>/rss.xml</code>、<code>/sitemap.xml</code>、<code>/robots.txt</code> 这些端点，其余一律归 notfound（与状态码无关）。<br>
  · 爬虫细分为：<code>search_bot</code>（搜索引擎）、<code>ai_bot</code>（GPTBot / ClaudeBot / Perplexity 等）、
  <code>scanner</code>（扫描器）、<code>datacenter</code>（机房 AS 的伪装浏览器流量）、
  <code>notfound</code>、<code>empty_ua</code>、<code>http_*</code>、<code>bot</code>。<br>
  · 分类字段 2026-09-20 才加。之前的旧数据没有分类，历史「真人」里仍含扫描器噪音。
</div>
<script>
const RANGES=[["1h","1 小时"],["6h","6 小时"],["24h","24 小时"],["7d","7 天"],["30d","30 天"],["90d","90 天"]];
const fmt = n => (n||0).toLocaleString('en-US');
const pct = (a,b) => b ? (a/b*100).toFixed(1)+'%' : '—';
const esc = s => String(s==null?'':s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
let CURRENT = new URL(location.href).searchParams.get('range') || '7d';

function renderRanges(){
  document.getElementById('ranges').innerHTML = RANGES.map(([k,l]) =>
    '<button class="'+(k===CURRENT?'on':'')+'" data-r="'+k+'">'+l+'</button>').join('');
  document.querySelectorAll('.ranges button').forEach(b => b.onclick = () => {
    CURRENT = b.dataset.r;
    const u = new URL(location.href); u.searchParams.set('range', CURRENT);
    history.replaceState(null,'',u);
    renderRanges(); load();
  });
}

function bar(v, max, cls){ return '<span class="bar '+(cls||'')+'" style="width:'+Math.max(2,Math.round(v/max*54))+'px"></span>'; }

function table(rows, cols, drill){
  let h = '<table><tr>' + cols.map(c=>'<th'+(c.n?' class="n"':'')+'>'+c.t+'</th>').join('') + '</tr>';
  h += rows.map(r => {
    const d = drill ? drill(r) : null;
    const attrs = d ? ' class="rowlink" data-src="'+d.src+'" data-dim="'+d.dim+'" data-val="'+esc(d.val)+'"' : '';
    return '<tr'+attrs+'>' + cols.map(c=>'<td'+(c.n?' class="n"':'')+'>'+c.f(r)+'</td>').join('') + '</tr>';
  }).join('');
  return h + '</table>';
}

// ── 下钻详情 ──
function detailTable(rows, src){
  if(!rows.length) return '<div class="muted">这个条件下没有记录</div>';
  const isBot = src === 'bot';
  const cols = [
    {t:'时间', f:r=>'<span class="muted">'+esc(String(r.timestamp).slice(0,19))+'</span>'},
    {t:'路径', f:r=>'<span style="word-break:break-all">'+esc(r.path)+'</span>'},
    {t:'国家', f:r=>esc(r.country||'—')},
  ];
  if(!isBot) cols.push({t:'城市', f:r=>esc(r.city||'—')});
  cols.push({t:'IP / 标识', f:r=>{
    if(r.oldIp) return '<span title="明文 IP（2026-09-20 之前的旧数据）">'+esc(r.oldIp)+'</span>';
    if(r.ipHash) return '<span class="muted" title="加盐 SHA-256 前 8 字节，不能反推 IP">'+esc(r.ipHash)+'</span>';
    return '—';
  }});
  cols.push({t:'AS 组织', f:r=>'<span class="muted">'+esc(r.asOrg||'—')+'</span>'});
  if(!isBot) cols.push({t:'来源', f:r=>'<span class="muted" style="word-break:break-all">'+esc(r.referer||'（直接访问）')+'</span>'});
  cols.push({t:'状态', n:true, f:r=>esc(r.status||'—')});
  cols.push({t:'User-Agent', f:r=>'<span class="muted" style="word-break:break-all;font-size:11px">'+esc(r.ua)+'</span>'});
  return '<table>' + '<tr>' + cols.map(c=>'<th'+(c.n?' class="n"':'')+'>'+c.t+'</th>').join('') + '</tr>' +
    rows.map(r=>'<tr>'+cols.map(c=>'<td'+(c.n?' class="n"':'')+'>'+c.f(r)+'</td>').join('')+'</tr>').join('') + '</table>';
}

// 当前下钻状态（翻页时复用）
let DRILL = null;
const PAGE_SIZES = [50, 100, 200];

function pager(state){
  const { page, totalPages, pageSize, total } = state;
  // 页码窗口：当前页前后各 2 页
  const nums = [];
  const from = Math.max(1, page - 2), to = Math.min(totalPages, page + 2);
  for (let i = from; i <= to; i++) nums.push(i);
  const mk = (label, p, dis, on) =>
    '<button class="pg' + (on ? ' on' : '') + '"' + (dis ? ' disabled' : '') + ' data-p="' + p + '">' + label + '</button>';
  let h = '<div class="pager">';
  h += '<span class="muted">共 ' + fmt(total) + ' 条 · 第 ' + page + ' / ' + totalPages + ' 页</span>';
  h += '<span class="pg-group">';
  h += mk('‹ 上一页', page - 1, page <= 1, false);
  if (from > 1) { h += mk('1', 1, false, page === 1); if (from > 2) h += '<span class="muted">…</span>'; }
  for (const n of nums) h += mk(String(n), n, false, n === page);
  if (to < totalPages) { if (to < totalPages - 1) h += '<span class="muted">…</span>'; h += mk(String(totalPages), totalPages, false, page === totalPages); }
  h += mk('下一页 ›', page + 1, page >= totalPages, false);
  h += '</span>';
  h += '<span class="pg-group"><span class="muted">每页</span>';
  for (const ps of PAGE_SIZES) h += '<button class="pg' + (ps === pageSize ? ' on' : '') + '" data-ps="' + ps + '">' + ps + '</button>';
  h += '</span>';
  h += '</div>';
  return h;
}

async function drill(src, dim, val, page, pageSize){
  const box = document.getElementById('detail');
  if(DRILL && (DRILL.src !== src || DRILL.dim !== dim || DRILL.val !== val)){
    page = 1; // 换分组时回到第一页
  }
  const ps = pageSize || (DRILL && DRILL.pageSize) || 50;
  DRILL = { src, dim, val, page: page || 1, pageSize: ps };
  box.style.display = 'block';
  box.innerHTML = '<h2>明细：' + esc(val) + ' <span class="muted">加载中…</span></h2>';
  if(!page) box.scrollIntoView({behavior:'smooth', block:'start'});
  try {
    const q = '/api/detail?range=' + encodeURIComponent(CURRENT) +
      '&src=' + esc(src) + '&dim=' + esc(dim) +
      '&val=' + encodeURIComponent(val) +
      '&page=' + DRILL.page + '&pageSize=' + DRILL.pageSize;
    const res = await fetch(q);
    if(!res.ok) throw new Error('HTTP ' + res.status + ' ' + (await res.text()).slice(0,160));
    const d = await res.json();
    const rows = d.data || [];
    const plain = rows.filter(r => r.oldIp).length;
    box.innerHTML =
      '<div style="display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap">' +
        '<h2 style="margin:0">明细：' + esc(val) + ' <span class="muted">（' + esc(src === 'bot' ? '爬虫' : '真人') + '）</span></h2>' +
        '<button id="detail-close" class="pg">收起</button>' +
      '</div>' +
      '<div class="muted" style="margin:8px 0 4px">' +
        (plain ? ('本页 <b>' + plain + '</b> 条是 2026-09-20 之前的旧数据，带<b>明文 IP</b>；其余只有哈希标识。')
               : '本页记录都只有加盐哈希标识（不能反推 IP）—— 明文 IP 已于 2026-09-20 停止写入。') +
      '</div>' +
      pager(d) + detailTable(rows, src) + pager(d);
    const btn = document.getElementById('detail-close');
    if(btn) btn.onclick = () => { box.style.display='none'; box.innerHTML=''; DRILL = null; };
  } catch(e) {
    box.innerHTML = '<h2>明细：' + esc(val) + '</h2><div class="err">加载失败：' + esc(String(e.message||e)) + '</div>';
  }
}
function chart(human, bot){
  const keys = [...new Set([...human.map(x=>x.t), ...bot.map(x=>x.t)])].sort();
  const hm = new Map(human.map(x=>[x.t, Number(x.n)]));
  const bm = new Map(bot.map(x=>[x.t, Number(x.n)]));
  const max = Math.max(1, ...keys.map(k => Math.max(hm.get(k)||0, bm.get(k)||0)));
  const bars = keys.map(k => {
    const hh = Math.round((hm.get(k)||0)/max*105), bb = Math.round((bm.get(k)||0)/max*105);
    return '<i style="height:'+Math.max(2,hh)+'px"></i><i class="bot" style="height:'+Math.max(2,bb)+'px"></i>';
  }).join('');
  return '<div class="chart">'+bars+'</div>' +
    '<div class="legend"><span><span class="dot" style="background:var(--coral)"></span>真人</span>' +
    '<span><span class="dot" style="background:var(--sub);opacity:.4"></span>爬虫</span>' +
    '<span>'+esc(keys[0]||'')+' → '+esc(keys[keys.length-1]||'')+'</span></div>';
}

function contentAnalysis(postHuman, postBot){
  const h = new Map(postHuman.map(x=>[x.path, Number(x.n)]));
  const b = new Map(postBot.map(x=>[x.path, Number(x.n)]));
  const paths = [...new Set([...h.keys(), ...b.keys()])];
  const rows = paths.map(p => ({ p, h: h.get(p)||0, b: b.get(p)||0 }));
  // 阈值按这个站的量级定（7 天全站真人 PV 只有几千，单篇 1~5 次是常态）
  const crawledNoRead = rows.filter(r => r.b >= 3 && r.h === 0).sort((x,y)=>y.b-x.b);
  const readNoCrawl = rows.filter(r => r.h >= 2 && r.b === 0).sort((x,y)=>y.h-x.h);
  const topRead = rows.filter(r => r.h > 0).sort((x,y)=>y.h-x.h).slice(0,15);
  // 注意：这段内联脚本最终会嵌进 Worker 的模板字符串里，
  // 不能用带反斜杠转义的正则（反斜杠会被外层模板字符串吃掉，变成非法正则）。
  const strip = p => (p.indexOf('/posts/') === 0 ? p.slice(7) : p);
  const decode = p => { try { return decodeURIComponent(p); } catch (e) { return p; } };
  const short = p => { const t = decode(strip(p)); return t.length > 46 ? t.slice(0,46)+'…' : t; };

  let html = '<div class="card full"><h2>内容优化建议 <span class="muted">（按文章对比「真人阅读」和「爬虫抓取」）</span></h2>';

  html += '<div style="margin-bottom:14px"><b style="font-size:12px">被爬虫抓了很多次、但几乎没人读</b> ' +
    '<span class="tag warn">标题 / 摘要 / 首屏可能需要改</span><table>' +
    table(crawledNoRead.slice(0,8), [
      {t:'文章路径', f:r=>'<span class="muted">'+esc(short(r.p))+'</span>'},
      {t:'真人阅读', n:true, f:r=>fmt(r.h)},
      {t:'爬虫抓取', n:true, f:r=>fmt(r.b)},
    ]) + '</table>' + (crawledNoRead.length? '' : '<div class="muted">（本次区间内没有这类文章 —— 试着把时间范围调大到 30 天或 90 天）</div>') + '</div>';

  html += '<div style="margin-bottom:14px"><b style="font-size:12px">有人读、但爬虫抓得少</b> ' +
    '<span class="tag warn">内链不足 / 索引没跟上</span><table>' +
    table(readNoCrawl.slice(0,8), [
      {t:'文章路径', f:r=>'<span class="muted">'+esc(short(r.p))+'</span>'},
      {t:'真人阅读', n:true, f:r=>fmt(r.h)},
      {t:'爬虫抓取', n:true, f:r=>fmt(r.b)},
    ]) + '</table>' + (readNoCrawl.length? '' : '<div class="muted">（本次区间内没有这类文章 —— 试着把时间范围调大到 30 天或 90 天）</div>') + '</div>';

  html += '<div><b style="font-size:12px">阅读量最高的文章</b><table>' +
    table(topRead, [
      {t:'文章路径', f:r=>'<span class="muted">'+esc(short(r.p))+'</span>'+bar(r.h, Math.max(1,topRead[0]?topRead[0].h:1))},
      {t:'真人阅读', n:true, f:r=>fmt(r.h)},
      {t:'爬虫抓取', n:true, f:r=>fmt(r.b)},
    ]) + '</table></div>';

  html += '</div>';
  return html;
}

async function load(){
  const app = document.getElementById('app');
  const meta = document.getElementById('meta');
  app.innerHTML = '<div class="muted">加载中…</div>';
  try {
    const res = await fetch('/api/stats?range='+encodeURIComponent(CURRENT));
    if (!res.ok) throw new Error('HTTP '+res.status+' '+(await res.text()).slice(0,140));
    const d = await res.json();
    const total = d.human.pv + d.bots.pv;
    meta.textContent = d.rangeLabel + ' · Cloudflare Analytics Engine · ' + new Date().toLocaleString('zh-CN');

    let html = '<div class="kpis">' + [
      ['真人页面浏览', fmt(d.human.pv), 'PV', 'coral'],
      ['真人访客', fmt(d.human.uv), '按 IP 哈希去重', 'teal'],
      ['RSS 订阅者', fmt(d.feed.pv), '阅读器请求，不算爬虫', 'blue'],
      ['爬虫 / 扫描', fmt(d.bots.pv), '占全部请求 ' + pct(d.bots.pv, total), 'amber'],
    ].map(([k,v,s,c]) => '<div class="kpi '+c+'"><div class="k">'+k+'</div><div class="v">'+v+'</div><div class="s">'+s+'</div></div>').join('') + '</div>';

    html += '<div class="grid">';
    html += '<div class="card full"><h2>趋势（按'+(d.bucket==='hour'?'小时':'天')+'）</h2>' + chart(d.trendHuman, d.trendBot) + '</div>';

    const mk = Math.max(1, ...d.kinds.map(x=>Number(x.n)));
    html += '<div class="card"><h2>爬虫 / 扫描器分类 <span class="muted">点行看明细</span></h2>' + table(d.kinds, [
      {t:'类型', f:x=>esc(x.botType)+bar(Number(x.n),mk)},
      {t:'kind', f:x=>'<span class="muted">'+esc(x.kind||'（旧数据无分类）')+'</span>'},
      {t:'次数', n:true, f:x=>fmt(Number(x.n))},
    ], x=>x.botType ? ({src:'bot', dim:'botType', val:x.botType}) : null) + '</div>';

    const mp = Math.max(1, ...d.pages.map(x=>Number(x.n)));
    html += '<div class="card"><h2>真人访问最多的页面 <span class="muted">点行看明细</span></h2>' + table(d.pages, [
      {t:'路径', f:x=>'<span class="muted" style="word-break:break-all">'+esc(x.path)+'</span>'+bar(Number(x.n),mp)},
      {t:'PV', n:true, f:x=>fmt(Number(x.n))},
    ], x=>({src:'human', dim:'path', val:x.path})) + '</div>';

    const mc = Math.max(1, ...d.countries.map(x=>Number(x.n)));
    html += '<div class="card"><h2>真人来源国家 <span class="muted">点行看明细</span></h2>' + table(d.countries, [
      {t:'国家', f:x=>esc(x.country||'（未知）')+bar(Number(x.n),mc,'teal')},
      {t:'PV', n:true, f:x=>fmt(Number(x.n))},
    ], x=>({src:'human', dim:'country', val:x.country})) + '</div>';

    const md = Math.max(1, ...d.devices.map(x=>Number(x.n)));
    html += '<div class="card"><h2>真人设备 / 系统 <span class="muted">点行看明细</span></h2>' + table(d.devices, [
      {t:'设备', f:x=>esc(x.device)+bar(Number(x.n),md,'blue')},
      {t:'PV', n:true, f:x=>fmt(Number(x.n))},
    ], x=>({src:'human', dim:'device', val:x.device})) + '</div>';

    const mr = Math.max(1, ...d.referers.map(x=>Number(x.n)));
    html += '<div class="card"><h2>真人来源（Referer） <span class="muted">点行看明细</span></h2>' + table(d.referers, [
      {t:'来源', f:x=>'<span class="muted" style="word-break:break-all">'+esc(x.referer||'（直接访问 / 无来源）')+'</span>'+bar(Number(x.n),mr,'teal')},
      {t:'PV', n:true, f:x=>fmt(Number(x.n))},
    ], x=>({src:'human', dim:'referer', val:x.referer})) + '</div>';

    html += contentAnalysis(d.postHuman, d.postBot);
    html += '</div>';
    app.innerHTML = html;
  } catch (e) {
    app.innerHTML = '<div class="err">加载失败：' + esc(String(e.message||e)) + '</div>';
  }
}

document.getElementById('app').addEventListener('click', ev => {
  const tr = ev.target.closest('tr.rowlink');
  if(tr) return drill(tr.dataset.src, tr.dataset.dim, tr.dataset.val);
});

// 下钻面板里的翻页按钮
document.getElementById('detail').addEventListener('click', ev => {
  const b = ev.target.closest('button.pg');
  if(!b || !DRILL) return;
  if(b.dataset.p) return drill(DRILL.src, DRILL.dim, DRILL.val, Number(b.dataset.p), DRILL.pageSize);
  if(b.dataset.ps) return drill(DRILL.src, DRILL.dim, DRILL.val, 1, Number(b.dataset.ps));
});
renderRanges();
load();
</script>
</body></html>`;

export default {
  async fetch(request, env) {
    if (!checkAuth(request, env)) return unauthorized();

    const url = new URL(request.url);

    if (url.pathname === "/api/detail") {
      try {
        const data = await detail(
          env,
          url.searchParams.get("range") || DEFAULT_RANGE,
          url.searchParams.get("src") || "bot",
          url.searchParams.get("dim") || "",
          url.searchParams.get("val") || "",
          url.searchParams.get("page"),
          url.searchParams.get("pageSize"),
        );
        return Response.json(data, {
          headers: { "Cache-Control": "no-store", "X-Robots-Tag": "noindex" },
        });
      } catch (e) {
        return new Response(String(e.message || e), { status: 400 });
      }
    }

    if (url.pathname === "/api/stats") {
      try {
        const data = await collect(env, url.searchParams.get("range") || DEFAULT_RANGE);
        return Response.json(data, {
          headers: { "Cache-Control": "no-store", "X-Robots-Tag": "noindex" },
        });
      } catch (e) {
        return new Response(String(e.message || e), { status: 500 });
      }
    }

    return new Response(PAGE, {
      headers: {
        "Content-Type": "text/html;charset=UTF-8",
        "Cache-Control": "no-store",
        "X-Robots-Tag": "noindex,nofollow",
      },
    });
  },
};
