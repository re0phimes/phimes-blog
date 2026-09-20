/**
 * site-analytics —— 站点访问统计（真人 / 爬虫 分流）
 *
 * 挂在 zone 路由：phimes.top/*、*.phimes.top/*
 * （views.phimes.top/* 与 stats.phimes.top/* 有更具体的路由，不会走这里）
 *
 * ── 判定逻辑（v3）────────────────────────────────────────────
 * 顺序很重要，从上到下第一个命中的决定归类：
 *
 *   ① 路径不属于站点内容（正向白名单）      → notfound
 *   ② 状态码不是 200 / 304                → http_<status>
 *   ③ UA 命中扫描器特征                   → scanner
 *   ④ 路径命中探针特征                     → scanner
 *   ⑤ UA 为空或过短                       → empty_ua
 *   ⑥ UA 命中 RSS 阅读器特征               → feed   （真人数据集）
 *   ⑦ UA 命中搜索引擎特征                  → search_bot
 *   ⑧ UA 命中 AI 爬虫特征                  → ai_bot
 *   ⑨ UA 命中其他爬虫特征                  → bot
 *   ⑩ IP 命中已知爬虫网段                  → bot
 *   ⑪ AS 组织名是机房                      → datacenter
 *   ⑫ cf.botManagement.score < 30         → bot
 *   ⑬ 以上都不匹配                         → browser（真人数据集）
 *
 * ── 为什么 ① 要用正向白名单 ──────────────────────────────────
 * 原来看状态码，只排除了 404。但浏览器缓存过一个 404 页面后再访问，
 * Cloudflare 会回 304 Not Modified，于是绕过了判定被算成真人
 * （实测 /NCWHtoBSD.html、/embedding_visulization.html 这些不存在的路径
 *   就是这样混进来的）。
 * 改成正向白名单后，不在内容前缀里的一律归 notfound，与状态码无关。
 *
 * ── blob 布局 ────────────────────────────────────────────────
 * site_visits: 1 path | 2 ipHash | 3 ""(原明文IP,已停写) | 4 referer
 *              5 country | 6 city | 7 hostname | 8 ua | 9 clientClass
 *              10 status | 11 asOrganization
 * bot_visits:  1 path | 2 ""(原明文IP,已停写) | 3 ua | 4 hostname | 5 country
 *              6 ipHash | 7 botType | 8 kind | 9 status | 10 asOrganization
 */

// ── 站点内容的路径前缀（正向白名单）──
// 这个站的内容全部在 /、/posts/、/pages/、/page 下；
// 其余路径（/.env、/wp-login.php、/about.html、/*/HEAD …）都是扫描或失效链接。
const CONTENT_EXACT = new Set(["/", "/page", "/redirect", "/redirect/"]);
const CONTENT_PREFIXES = ["/posts/", "/pages/", "/page/"];

// 订阅端点：RSS 阅读器取的是这些，属于真实订阅行为，不能当成 notfound
const FEED_ENDPOINTS = new Set(["/rss.xml", "/feed.xml", "/atom.xml"]);

// 爬虫端点：这些是「谁在读我的站点」最有价值的信号，
// 不能因为不是内容页就丢进 notfound，要交给 UA 分类去认。
const CRAWLER_ENDPOINTS = new Set([
  "/robots.txt",
  "/sitemap.xml",
  "/llms.txt",
  "/index.json",
  "/manifest.webmanifest",
]);

// 完全不需要统计的静态资源（两个数据集都不记，保证口径一致）
const STATIC_EXTENSIONS = [
  ".js",
  ".mjs",
  ".css",
  ".png",
  ".jpg",
  ".jpeg",
  ".gif",
  ".svg",
  ".ico",
  ".woff",
  ".woff2",
  ".ttf",
  ".eot",
  ".map",
  ".webp",
  ".avif",
];
const SKIP_PREFIXES = ["/api/", "/assets/", "/covers/", "/images/", "/fonts/", "/_next/"];

// 扫描器 / 资产测绘平台
const SCANNER_UA_PATTERNS = [
  /l9scan/i,
  /leakix/i,
  /tlm-audit/i,
  /crusader/i,
  /mozlila/i, // 伪造 Chrome UA 时把 Mozilla 拼错了
  /nuclei/i,
  /masscan/i,
  /zgrab/i,
  /sqlmap/i,
  /nikto/i,
  /wpscan/i,
  /dirbuster/i,
  /gobuster/i,
  /feroxbuster/i,
  /nmap/i,
  /acunetix/i,
  /netsparker/i,
  /qualys/i,
  /censys/i,
  /shodan/i,
  /internet-measurement/i,
  /paloalto/i,
  /expanse/i,
  /bitsight/i,
  /recyber/i,
  /alphastrike/i,
  /securityscorecard/i,
  /scanner\b/i,
  /vuln/i,
  /\bprobe\b/i,
];

// 本站不存在的路径特征（命中即扫描，不再依赖状态码）
const PROBE_PATH_PATTERNS = [
  /^\/\.env/i,
  /^\/\.git/i,
  /^\/\.aws/i,
  /^\/\.ssh/i,
  /^\/wp-/i,
  /^\/wordpress/i,
  /^\/xmlrpc/i,
  /^\/phpmyadmin/i,
  /^\/pma/i,
  /^\/cgi-bin/i,
  /^\/vendor\//i,
  /^\/node_modules\//i,
  /\/\.(php|asp|aspx|jsp|cgi|env|bak|old|sql|zip|tar|gz|ini|log|yml|yaml)$/i,
];

// RSS / Atom 阅读器 —— 真实订阅者，归入真人数据集。
// 注意：不要用 /rss/ 或 /feed/ 这种宽泛匹配 —— feedparser 这类抓取库
// 名字里带 feed，会被误判成真人。
const FEED_UA_PATTERNS = [
  /scourrss/i,
  /feedly/i,
  /inoreader/i,
  /newsblur/i,
  /the old reader/i,
  /netnewswire/i,
  /reeder/i,
  /miniflux/i,
  /tiny tiny rss/i,
  /tt-rss/i,
  /rssowl/i,
  /quiterss/i,
  /akregator/i,
  /liferea/i,
  /feedreader/i,
  /feedbin/i,
  /newsboat/i,
  /omnivore/i,
  /freshreader/i,
  /rssbandit/i,
  / Vienna\//i,
];

// 搜索引擎爬虫
const SEARCH_BOT_MAP = [
  [/googlebot/i, "googlebot"],
  [/bingbot/i, "bingbot"],
  [/baiduspider|baidu\//i, "baiduspider"],
  [/yandex/i, "yandexbot"],
  [/duckduckbot/i, "duckduckbot"],
  [/sogou/i, "sogou"],
  [/360spider/i, "360spider"],
  [/applebot/i, "applebot"],
  [/petalbot/i, "petalbot"],
  [/exabot/i, "exabot"],
  [/yisouspider/i, "yisouspider"],
];

// AI / Agent 爬虫
const AI_BOT_MAP = [
  [/oai-search/i, "oai_search"],
  [/chatgpt-user/i, "chatgpt_user"],
  [/gptbot|chatgpt/i, "gptbot"],
  [/claudebot|claude-web/i, "claudebot"],
  [/anthropic/i, "anthropic"],
  [/perplexity/i, "perplexity"],
  [/ccbot/i, "ccbot"],
  [/bytespider/i, "bytespider"],
  [/deepseek/i, "deepseek"],
  [/grok/i, "grok"],
  [/cohere/i, "cohere"],
  [/google-cloudvertex/i, "google_vertex"],
  [/moonshot|kimi/i, "moonshot"],
  [/yibot/i, "yibot"],
  [/shapbot/i, "shap"],
  [/reflectionbot/i, "reflection"],
  [/youbot/i, "youbot"],
  [/amazonbot|amzn-search|amazon-search/i, "amazonbot"],
];

// 其他爬虫 / 命令行工具
const OTHER_BOT_MAP = [
  [/semrush/i, "semrushbot"],
  [/ahrefs/i, "ahrefsbot"],
  [/mj12bot/i, "mj12bot"],
  [/dotbot/i, "dotbot"],
  [/dataforseo/i, "dataforseo"],
  [/seranking/i, "seranking"],
  [/js_?crawler|jscrawler/i, "jscrawler"],
  [/headlesschrome/i, "headless_chrome"],
  [/phantomjs/i, "phantomjs"],
  [/selenium/i, "selenium"],
  [/puppeteer/i, "puppeteer"],
  [/playwright/i, "playwright"],
  [/scrapy/i, "scrapy"],
  [/nutch/i, "nutch"],
  [/curl/i, "curl"],
  [/wget/i, "wget"],
  [/httpie/i, "httpie"],
  [/python-requests/i, "python_requests"],
  [/feedparser/i, "feedparser"],
  [/axios/i, "axios"],
  [/node-fetch/i, "node_fetch"],
  [/go-http-client/i, "go_http"],
  [/java\//i, "java_client"],
  [/libwww/i, "libwww"],
  [/archive\.org|ia_archiver/i, "archive_org"],
  [/facebookexternalhit|facebookbot|facebot/i, "facebookbot"],
  [/twitterbot/i, "twitterbot"],
  [/linkedinbot/i, "linkedinbot"],
  [/telegrambot/i, "telegrambot"],
  [/discordbot/i, "discordbot"],
  [/slackbot/i, "slackbot"],
  [/whatsapp/i, "whatsapp"],
  [/pinterest/i, "pinterest"],
  [/embedly/i, "embedly"],
  [/outbrain/i, "outbrain"],
  [/quora link/i, "quora"],
  [/flipboard/i, "flipboard"],
];

// 用户 UA 里出现这些词就当成爬虫（兜底）
const BOT_UA_HINT = /bot|crawl|spider|slurp|mediapartners/i;

const BOT_IP_PREFIXES = [
  "66.249.", // Googlebot
  "64.233.",
  "72.14.",
  "209.85.",
  "216.239.",
  "40.77.", // Bingbot
  "157.55.",
  "207.46.",
  "13.66.",
  "77.88.", // Yandex
  "141.8.",
  "5.255.",
  "180.76.", // Baidu
  "220.181.",
  "114.119.", // Bytespider
];

// 云厂商 / 机房 AS —— 真人不应该从这些地方访问。
// 用 asOrganization（免费套餐就能读到），不像 botManagement 需要付费。
const HOSTING_ASN = new RegExp(
  [
    "DigitalOcean",
    "Amazon|AWS",
    "Google",
    "Microsoft|Azure",
    "OVH",
    "Hetzner",
    "Linode|Akamai",
    "Vultr",
    "Contabo",
    "Scaleway",
    "Leaseweb",
    "M247",
    "Oracle",
    "Alibaba|Aliyun",
    "Tencent",
    "Cloudflare",
    "Cloudie|HostHatch|BuyVM|RackNerd|ColoCrossing",
    "Choopa|Constant",
    "Serverion|Datacamp|DataCamp",
    "Zenlayer|UCloud|QingCloud",
    "Kagoya|Sakura",
    "Online SAS|Online\\.net",
  ].join("|"),
  "i",
);

const IP_HASH_SALT_FALLBACK = "phimes-analytics-v2-stable";

function matchMap(map, ua) {
  for (const [re, name] of map) if (re.test(ua)) return name;
  return null;
}

function isContentPath(pathname) {
  if (CONTENT_EXACT.has(pathname)) return true;
  return CONTENT_PREFIXES.some((p) => pathname.startsWith(p));
}

function isStaticAsset(pathname) {
  if (SKIP_PREFIXES.some((p) => pathname.startsWith(p))) return true;
  return STATIC_EXTENSIONS.some((ext) => pathname.endsWith(ext));
}

function isProbePath(pathname) {
  return PROBE_PATH_PATTERNS.some((re) => re.test(pathname));
}

async function hashIP(ip, salt) {
  const data = new TextEncoder().encode(ip + salt);
  const hash = await crypto.subtle.digest("SHA-256", data);
  const arr = new Uint8Array(hash);
  return Array.from(arr.slice(0, 8))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

function classify(request, response, pathname) {
  const ua = request.headers.get("user-agent") || "";
  const ip = request.headers.get("cf-connecting-ip") || "";
  const asOrg = request.cf?.asOrganization || "";

  const isFeedEndpoint = FEED_ENDPOINTS.has(pathname);
  const isCrawlerEndpoint = CRAWLER_ENDPOINTS.has(pathname);

  // ① 路径不属于站点内容 —— 与状态码无关。
  //    这一步同时覆盖了「不存在的路径」和「缓存过的 404 返回 304」两种情况。
  //    /rss.xml、/sitemap.xml 这些要放行，交给后面的 UA 分类处理。
  if (isProbePath(pathname)) {
    return { dataset: "bot", kind: "scanner", botType: "probe_path" };
  }
  if (!isContentPath(pathname) && !isFeedEndpoint && !isCrawlerEndpoint) {
    return { dataset: "bot", kind: "notfound", botType: "notfound" };
  }

  // ② 非成功响应不算页面浏览（404 前面已覆盖，这里管 403 / 429 / 5xx）
  if (response.status !== 200 && response.status !== 304) {
    return { dataset: "bot", kind: `http_${response.status}`, botType: `http_${response.status}` };
  }

  // ③ 扫描器
  if (SCANNER_UA_PATTERNS.some((re) => re.test(ua))) {
    return { dataset: "bot", kind: "scanner", botType: "scanner_ua" };
  }

  // ④ UA 为空 / 过短
  if (!ua || ua.length < 10) {
    return { dataset: "bot", kind: "empty_ua", botType: "empty_ua" };
  }

  // ⑤ RSS 阅读器 —— 真实订阅者。
  //    放在爬虫判定之前，但用的是收窄后的白名单，不会把 feedparser 这类抓取库算进来。
  if (FEED_UA_PATTERNS.some((re) => re.test(ua))) {
    return { dataset: "human", clientClass: "feed" };
  }

  // ⑥ 搜索引擎
  const search = matchMap(SEARCH_BOT_MAP, ua);
  if (search) return { dataset: "bot", kind: "search_bot", botType: search };

  // ⑦ AI / Agent 爬虫
  const ai = matchMap(AI_BOT_MAP, ua);
  if (ai) return { dataset: "bot", kind: "ai_bot", botType: ai };

  // ⑧ 其他爬虫 / 命令行工具
  const other = matchMap(OTHER_BOT_MAP, ua);
  if (other) return { dataset: "bot", kind: "bot", botType: other };

  // ⑨ UA 里带 bot/crawl/spider 字样的兜底
  if (BOT_UA_HINT.test(ua)) {
    const m = ua.match(/([a-z0-9_-]+)(?:bot|crawler|spider)/i);
    const name = m && m[1] && m[1].length > 1 ? `${m[1].toLowerCase()}_bot` : "unknown_bot";
    return { dataset: "bot", kind: "bot", botType: name };
  }

  // ⑩ 已知爬虫网段
  if (BOT_IP_PREFIXES.some((prefix) => ip.startsWith(prefix))) {
    return { dataset: "bot", kind: "bot", botType: "bot_ip" };
  }

  // ⑪ 机房 AS —— UA 伪装成浏览器的 VPS 爬虫主要靠这一步兜住
  if (asOrg && HOSTING_ASN.test(asOrg)) {
    return { dataset: "bot", kind: "datacenter", botType: `dc:${asOrg.slice(0, 40)}` };
  }

  // ⑫ Cloudflare bot score（免费套餐下不注入，付费后自动生效）
  const botScore = request.cf?.botManagement?.score;
  if (botScore !== undefined && botScore < 30) {
    return { dataset: "bot", kind: "bot", botType: "bot_score" };
  }

  // ⑬ 真人
  return { dataset: "human", clientClass: "browser" };
}

export default {
  async fetch(request, env, ctx) {
    const response = await fetch(request);

    if (request.method !== "GET") return response;

    const url = new URL(request.url);
    const pathname = url.pathname;

    // 图片 / JS / CSS 这些不记，保证「真人 vs 爬虫」两个数据集口径一致
    if (isStaticAsset(pathname)) return response;

    // 响应已经返回给访客了，统计放到之后异步做，不增加延迟
    ctx.waitUntil(
      (async () => {
        try {
          await record(request, env, response, url);
        } catch (e) {
          console.error(
            JSON.stringify({
              msg: "analytics write failed",
              path: pathname,
              error: String((e && e.message) || e),
            }),
          );
        }
      })(),
    );

    return response;
  },
};

async function record(request, env, response, url) {
  const pathname = url.pathname;
  const hostname = url.hostname;
  const ua = request.headers.get("user-agent") || "";
  const ip = request.headers.get("cf-connecting-ip") || "unknown";
  const country = request.cf?.country || "";
  const city = request.cf?.city || "";
  const asOrg = request.cf?.asOrganization || "";
  const status = String(response.status);
  // 盐从 secret 读，仓库源码里不再出现（旧的硬编码值只作兜底）
  const salt = env.IP_HASH_SALT || IP_HASH_SALT_FALLBACK;
  const ipHash = await hashIP(ip, salt);
  const index = [hostname];

  const verdict = classify(request, response, pathname);

  if (verdict.dataset === "human") {
    env.ANALYTICS.writeDataPoint({
      // blob3 原来是明文 IP，保留空位不写，避免破坏已有查询的列序
      blobs: [
        pathname,
        ipHash,
        "",
        request.headers.get("referer") || "",
        country,
        city,
        hostname,
        ua,
        verdict.clientClass,
        status,
        asOrg,
      ],
      doubles: [1],
      indexes: index,
    });
    return;
  }

  env.BOT_ANALYTICS.writeDataPoint({
    // blob2 原来是明文 IP，保留空位不写
    blobs: [
      pathname,
      "",
      ua,
      hostname,
      country,
      ipHash,
      verdict.botType,
      verdict.kind,
      status,
      asOrg,
    ],
    doubles: [1],
    indexes: index,
  });
}
