/**
 * analytics-dashboard —— 只读的站点统计看板
 *
 * 挂在 stats.phimes.top/* （比 site-analytics 的 *.phimes.top/* 更具体，优先级更高）
 *
 * 数据来源：Cloudflare Analytics Engine，两个数据集
 *   site_visits —— 真实读者（blob9 = 'browser' | 'feed'）
 *   bot_visits  —— 爬虫 / 扫描器 / 空 UA / 404 探测（blob8 = kind）
 *
 * 鉴权：HTTP Basic Auth，密码从环境变量 DASH_PASSWORD 读（Worker secret）。
 * AE 的只读 token 也从 secret 读，不会暴露到浏览器。
 *
 * 设计约束：
 * - 不引任何外部依赖（不加载 CDN 上的 JS/CSS），保证国内打开也快
 * - 只用 system font stack
 */

const RANGE_DAYS = 30;

function unauthorized() {
  return new Response("需要登录", {
    status: 401,
    headers: { "WWW-Authenticate": 'Basic realm="analytics", charset="UTF-8"' },
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
    const pass = idx >= 0 ? decoded.slice(idx + 1) : decoded;
    return pass === expected;
  } catch {
    return false;
  }
}

async function aeQuery(env, sql) {
  const accountId = env.CF_ACCOUNT_ID;
  const res = await fetch(
    `https://api.cloudflare.com/client/v4/accounts/${accountId}/analytics_engine/sql`,
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

// 真人 = 浏览器页面浏览；旧数据没有 blob9，用 != 'feed' 兼容
const HUMAN = `timestamp > NOW() - INTERVAL '${RANGE_DAYS}' DAY AND blob9 != 'feed'`;
const FEED = `timestamp > NOW() - INTERVAL '${RANGE_DAYS}' DAY AND blob9 = 'feed'`;
const BOTS = `timestamp > NOW() - INTERVAL '${RANGE_DAYS}' DAY`;

async function collect(env) {
  const [human, feed, bots, kinds, pages, countries, daily, scannerPaths] = await Promise.all([
    aeQuery(
      env,
      `SELECT count() AS pv, count(DISTINCT blob2) AS uv FROM site_visits WHERE ${HUMAN}`,
    ),
    aeQuery(env, `SELECT count() AS pv, count(DISTINCT blob2) AS uv FROM site_visits WHERE ${FEED}`),
    aeQuery(env, `SELECT count() AS n FROM bot_visits WHERE ${BOTS}`),
    aeQuery(
      env,
      `SELECT blob8 AS kind, blob7 AS botType, count() AS n FROM bot_visits WHERE ${BOTS} GROUP BY kind, botType ORDER BY n DESC LIMIT 25`,
    ),
    aeQuery(
      env,
      `SELECT blob1 AS path, count() AS n FROM site_visits WHERE ${HUMAN} GROUP BY path ORDER BY n DESC LIMIT 15`,
    ),
    aeQuery(
      env,
      `SELECT blob5 AS country, count() AS n FROM site_visits WHERE ${HUMAN} GROUP BY country ORDER BY n DESC LIMIT 10`,
    ),
    aeQuery(
      env,
      `SELECT toDate(timestamp) AS d, count() AS n FROM site_visits WHERE ${HUMAN} GROUP BY d ORDER BY d ASC`,
    ),
    aeQuery(
      env,
      `SELECT blob1 AS path, count() AS n FROM bot_visits WHERE ${BOTS} AND blob8 = 'scanner' GROUP BY path ORDER BY n DESC LIMIT 10`,
    ),
  ]);

  const num = (r, k) => Number((r.data && r.data[0] && r.data[0][k]) || 0);

  return {
    rangeDays: RANGE_DAYS,
    human: { pv: num(human, "pv"), uv: num(human, "uv") },
    feed: { pv: num(feed, "pv"), uv: num(feed, "uv") },
    bots: num(bots, "n"),
    kinds: kinds.data || [],
    pages: pages.data || [],
    countries: countries.data || [],
    daily: daily.data || [],
    scannerPaths: scannerPaths.data || [],
  };
}

const PAGE = `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<title>站点统计 · Phimes</title>
<style>
  :root{--bg:#faf9f5;--card:#fff;--line:#e6dfd8;--ink:#141413;--sub:#6b6a64;--coral:#cc785c;--teal:#2f8574;--amber:#b8791f}
  *{box-sizing:border-box}
  body{margin:0;padding:24px;background:var(--bg);color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
    font-size:14px;line-height:1.6}
  h1{font-size:20px;margin:0 0 4px}
  .meta{color:var(--sub);font-size:12px;margin-bottom:20px}
  .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:20px}
  .kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
  .kpi .k{color:var(--sub);font-size:12px}
  .kpi .v{font-size:26px;font-weight:600;letter-spacing:-.02em;margin-top:2px}
  .kpi .s{color:var(--sub);font-size:11px;margin-top:2px}
  .kpi.coral .v{color:var(--coral)} .kpi.teal .v{color:var(--teal)} .kpi.amber .v{color:var(--amber)}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
  .card h2{font-size:13px;margin:0 0 12px;color:var(--sub);font-weight:600;letter-spacing:.02em}
  table{width:100%;border-collapse:collapse;font-size:13px}
  td,th{padding:5px 0;border-bottom:1px solid var(--line);text-align:left}
  th{color:var(--sub);font-weight:500;font-size:11px}
  td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
  tr:last-child td{border-bottom:0}
  .bar{display:inline-block;height:6px;background:var(--coral);border-radius:3px;vertical-align:middle;margin-left:8px;opacity:.55}
  .trend{display:flex;align-items:flex-end;gap:2px;height:90px;margin-top:8px}
  .trend i{flex:1;background:var(--coral);border-radius:2px 2px 0 0;min-height:2px;opacity:.75}
  .note{color:var(--sub);font-size:11px;margin-top:20px;line-height:1.7}
  code{background:var(--card);border:1px solid var(--line);border-radius:4px;padding:1px 5px;font-size:11px}
  .err{background:#fff5f5;border:1px solid #e6c9c9;color:#a33;padding:12px;border-radius:8px}
</style></head><body>
<h1>站点统计</h1>
<div class="meta" id="meta">加载中…</div>
<div id="app"></div>
<div class="note">
  <b>口径说明</b><br>
  · <b>真人</b> = 返回 200 的页面浏览（静态资源不计）。RSS 阅读器单独记为「订阅者」。<br>
  · <b>爬虫/扫描</b> = 爬虫 UA、扫描器、空 UA、以及所有 404 探测。<br>
  · 分类字段（<code>blob8</code> kind / <code>blob9</code> clientClass）是 2026-09-20 才加的，
    之前的旧数据没有分类，所以历史「真人」里仍含扫描器噪音。<br>
  · 新数据已不再写入明文 IP，只保留加盐哈希用于 UV 去重。
</div>
<script>
const fmt = n => (n||0).toLocaleString('en-US');
function pct(a,b){ return b ? (a/b*100).toFixed(1)+'%' : '—'; }
function bars(rows, max){ return rows.map(r=>'<i style="height:'+Math.max(2,Math.round(r/max*90))+'px"></i>').join(''); }

(async () => {
  const app = document.getElementById('app');
  const meta = document.getElementById('meta');
  try {
    const res = await fetch('/api/stats');
    if (!res.ok) throw new Error('HTTP ' + res.status + ' ' + (await res.text()).slice(0,120));
    const d = await res.json();
    const total = d.human.pv + d.bots;
    meta.textContent = '最近 ' + d.rangeDays + ' 天 · 数据来源 Cloudflare Analytics Engine · 更新于 ' + new Date().toLocaleString('zh-CN');

    const kpis = [
      ['真人页面浏览', fmt(d.human.pv), 'PV', 'coral'],
      ['真人访客', fmt(d.human.uv), '按 IP 哈希去重（UV）', 'teal'],
      ['RSS 订阅者', fmt(d.feed.pv), '阅读器请求数，不算爬虫', ''],
      ['爬虫 / 扫描', fmt(d.bots), '占全部请求 ' + pct(d.bots, total), 'amber'],
    ];
    let html = '<div class="kpis">' + kpis.map(([k,v,s,c]) =>
      '<div class="kpi '+c+'"><div class="k">'+k+'</div><div class="v">'+v+'</div><div class="s">'+s+'</div></div>').join('') + '</div>';

    html += '<div class="grid">';

    // 爬虫类型
    const maxK = Math.max(1, ...d.kinds.map(x=>Number(x.n)));
    html += '<div class="card"><h2>爬虫 / 扫描器分类</h2><table><tr><th>类型</th><th>kind</th><th class="n">次数</th></tr>' +
      d.kinds.map(x => '<tr><td>'+x.botType+'<span class="bar" style="width:'+Math.round(Number(x.n)/maxK*60)+'px"></span></td><td style="color:var(--sub);font-size:11px">'+(x.kind||'（旧数据无分类）')+'</td><td class="n">'+fmt(Number(x.n))+'</td></tr>').join('') +
      '</table></div>';

    // Top 页面
    const maxP = Math.max(1, ...d.pages.map(x=>Number(x.n)));
    html += '<div class="card"><h2>真人访问最多的页面</h2><table><tr><th>路径</th><th class="n">PV</th></tr>' +
      d.pages.map(x => '<tr><td style="word-break:break-all">'+x.path+'<span class="bar" style="width:'+Math.round(Number(x.n)/maxP*50)+'px"></span></td><td class="n">'+fmt(Number(x.n))+'</td></tr>').join('') +
      '</table></div>';

    // 国家
    const maxC = Math.max(1, ...d.countries.map(x=>Number(x.n)));
    html += '<div class="card"><h2>真人来源国家</h2><table><tr><th>国家</th><th class="n">PV</th></tr>' +
      d.countries.map(x => '<tr><td>'+(x.country||'（未知）')+'<span class="bar" style="width:'+Math.round(Number(x.n)/maxC*50)+'px"></span></td><td class="n">'+fmt(Number(x.n))+'</td></tr>').join('') +
      '</table></div>';

    // 趋势
    const maxD = Math.max(1, ...d.daily.map(x=>Number(x.n)));
    html += '<div class="card"><h2>真人浏览趋势</h2><div class="trend">' + bars(d.daily.map(x=>Number(x.n)), maxD) + '</div>' +
      '<div style="color:var(--sub);font-size:11px;margin-top:6px">' + (d.daily[0]?.d||'') + ' → ' + (d.daily[d.daily.length-1]?.d||'') + '</div></table></div>';

    // 扫描路径
    if (d.scannerPaths.length) {
      const maxS = Math.max(1, ...d.scannerPaths.map(x=>Number(x.n)));
      html += '<div class="card"><h2>被扫描最多的路径（新数据）</h2><table><tr><th>路径</th><th class="n">次数</th></tr>' +
        d.scannerPaths.map(x => '<tr><td style="word-break:break-all">'+x.path+'<span class="bar" style="width:'+Math.round(Number(x.n)/maxS*50)+'px;background:var(--amber)"></span></td><td class="n">'+fmt(Number(x.n))+'</td></tr>').join('') +
        '</table></div>';
    }

    html += '</div>';
    app.innerHTML = html;
  } catch (e) {
    app.innerHTML = '<div class="err">加载失败：' + String(e.message||e).replace(/[<>]/g,'') + '</div>';
  }
})();
</script>
</body></html>`;

export default {
  async fetch(request, env) {
    if (!checkAuth(request, env)) return unauthorized();

    const url = new URL(request.url);

    if (url.pathname === "/api/stats") {
      try {
        const data = await collect(env);
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
