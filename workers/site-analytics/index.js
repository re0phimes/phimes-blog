/**
 * site-analytics —— 站点访问统计（真人 / 爬虫 分流）
 *
 * 挂在 zone 路由：phimes.top/*、*.phimes.top/*
 * （views.phimes.top/* 由 blog-page-views 的更具体路由接管，不会走这里）
 *
 * 数据写进两个 Analytics Engine 数据集：
 *   site_visits —— 真实读者（浏览器 + RSS 订阅客户端）
 *   bot_visits  —— 爬虫 / 扫描器 / 空 UA / 404 探测
 *
 * ── 相比上一版的改动 ────────────────────────────────────────────
 * 1. 【性能】改用 ctx.waitUntil()。原来每次请求都要 await 完 IP 哈希和写数据
 *    才返回响应，等于给全站每个请求都加了延迟。现在统计在响应之后异步做。
 * 2. 【准确性】用 HTTP 状态码判断。404 一律不算真人浏览 —— 这是「真人数据
 *    被扫描器污染 31%」的根因（/.env、/.git/config、/wp-login.php 这些都返回 404，
 *    但旧版只看 UA，就全算成真人了）。
 * 3. 【准确性】新增扫描器识别（l9scan / leakix / TLM-Audit / crusader / Mozlila 等
 *    伪造 UA 的漏网之鱼），以及探针路径识别。
 * 4. 【准确性】RSS 阅读器不再算爬虫 —— 它们是真实订阅者，归入 site_visits
 *    并用 clientClass='feed' 标记，可以单独统计也可以合并。
 * 5. 【一致性】两个数据集用同一套静态资源过滤，数字可以直接对比。
 * 6. 【隐私】不再写入明文 IP，只写加盐哈希（UV 去重仍然可用）。
 * 7. 【可观测】写数据失败会记日志，不再静默吞掉。
 *
 * ── blob 布局 ──────────────────────────────────────────────────
 * site_visits: 1 path | 2 ipHash | 3 ""(保留位,原为明文IP) | 4 referer
 *              5 country | 6 city | 7 hostname | 8 ua | 9 clientClass | 10 status
 * bot_visits:  1 path | 2 ""(保留位) | 3 ua | 4 hostname | 5 country
 *              6 ipHash | 7 botType | 8 kind | 9 status
 */

const BOT_UA_PATTERNS = [
  /bot/i,
  /crawl/i,
  /spider/i,
  /slurp/i,
  /mediapartners/i,
  /googlebot/i,
  /bingbot/i,
  /yandex/i,
  /baidu/i,
  /duckduckbot/i,
  /sogou/i,
  /exabot/i,
  /facebot/i,
  /ia_archiver/i,
  /semrush/i,
  /ahrefs/i,
  /mj12bot/i,
  /dotbot/i,
  /petalbot/i,
  /bytespider/i,
  /gptbot/i,
  /chatgpt/i,
  /claudebot/i,
  /anthropic/i,
  /ccbot/i,
  /dataforseo/i,
  /headlesschrome/i,
  /phantomjs/i,
  /selenium/i,
  /puppeteer/i,
  /playwright/i,
  /wget/i,
  /curl/i,
  /httpie/i,
  /python-requests/i,
  /axios/i,
  /node-fetch/i,
  /go-http-client/i,
  /java\//i,
  /libwww/i,
  /scrapy/i,
  /nutch/i,
  /archive\.org/i,
  /applebot/i,
  /twitterbot/i,
  /linkedinbot/i,
  /whatsapp/i,
  /telegrambot/i,
  /discordbot/i,
  /slackbot/i,
  /embedly/i,
  /quora link/i,
  /outbrain/i,
  /pinterest/i,
  /flipboard/i,
];

// 安全扫描器 / 资产测绘平台 —— 旧版漏掉的就是这一类，导致 8000+ 次被算成真人
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
  /scan/i,
  /audit/i,
  /probe/i,
  /\bchecker\b/i,
  /\bcrawler-check\b/i,
];

// RSS / Atom 阅读器 —— 真实订阅者，不该算爬虫
const FEED_UA_PATTERNS = [
  /scourrss/i,
  /feedly/i,
  /inoreader/i,
  /newsblur/i,
  /the old reader/i,
  /netnewswire/i,
  /reeder/i,
  /freshreader/i,
  /miniflux/i,
  /tiny tiny rss/i,
  /tt-rss/i,
  /rssowl/i,
  /quiteRSS/i,
  /akregator/i,
  /liferea/i,
  /feedreader/i,
  /feedbin/i,
  /newsboat/i,
  /omnivore/i,
  /zapier/i,
  /ifttt/i,
  /rss\b/i,
  /feed\b/i,
];

const BOT_IP_PREFIXES = [
  "66.249.", // Googlebot
  "64.233.", // Google
  "72.14.", // Google
  "209.85.", // Google
  "216.239.", // Google
  "40.77.", // Bingbot
  "157.55.", // Bingbot
  "207.46.", // Bingbot
  "13.66.", // Bingbot
  "77.88.", // Yandex
  "141.8.", // Yandex
  "180.76.", // Baidu
  "220.181.", // Baidu
  "5.255.", // Yandex
  "114.119.", // Bytespider
];

// 本站不存在的路径（不是 WordPress、没有 .env/.git），命中即为扫描探针
const PROBE_PATH_PATTERNS = [
  /^\/\.env/i,
  /^\/\.git/i,
  /^\/\.aws/i,
  /^\/\.ssh/i,
  /^\/\.well-known\/(?!security\.txt|change-password)/i,
  /^\/wp-/i,
  /^\/wordpress/i,
  /^\/xmlrpc/i,
  /^\/phpmyadmin/i,
  /^\/pma/i,
  /^\/admin/i,
  /^\/administrator/i,
  /^\/cgi-bin/i,
  /^\/vendor\//i,
  /^\/node_modules\//i,
  /^\/config/i,
  /^\/backup/i,
  /^\/db\./i,
  /^\/dump/i,
  /^\/sql/i,
  /\.(php|asp|aspx|jsp|cgi|env|bak|old|sql|zip|tar|gz|ini|log|yml|yaml)$/i,
];

const SKIP_PATH_PREFIXES = ["/api/", "/_next/"];
const SKIP_PATH_EXACT = [
  "/manifest.webmanifest",
  "/robots.txt",
  "/sitemap.xml",
  "/favicon.ico",
  "/llms.txt",
];
const SKIP_EXTENSIONS = [
  ".js",
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
  ".xml",
  ".json",
  ".txt",
];

const HASH_SALT = "phimes-analytics-v2-stable";

function classifyBot(ua) {
  const uaLower = ua.toLowerCase();
  if (uaLower.includes("googlebot")) return "googlebot";
  if (uaLower.includes("bingbot")) return "bingbot";
  if (uaLower.includes("baidu")) return "baiduspider";
  if (uaLower.includes("yandex")) return "yandexbot";
  if (uaLower.includes("duckduckbot")) return "duckduckbot";
  if (uaLower.includes("sogou")) return "sogou";
  if (uaLower.includes("360spider")) return "360spider";
  if (uaLower.includes("gptbot") || uaLower.includes("chatgpt")) return "gptbot";
  if (uaLower.includes("oai-search")) return "oai_search";
  if (uaLower.includes("claudebot")) return "claudebot";
  if (uaLower.includes("anthropic")) return "anthropic";
  if (uaLower.includes("perplexity")) return "perplexity";
  if (uaLower.includes("ccbot")) return "ccbot";
  if (uaLower.includes("deepseek")) return "deepseek";
  if (uaLower.includes("grok")) return "grok";
  if (uaLower.includes("cohere")) return "cohere";
  if (uaLower.includes("google-cloudvertex")) return "google_vertex";
  if (uaLower.includes("dataforseo")) return "dataforseo";
  if (uaLower.includes("bytespider")) return "bytespider";
  if (uaLower.includes("ahrefs")) return "ahrefsbot";
  if (uaLower.includes("semrush")) return "semrushbot";
  if (uaLower.includes("mj12bot")) return "mj12bot";
  if (uaLower.includes("dotbot")) return "dotbot";
  if (uaLower.includes("petalbot")) return "petalbot";
  if (uaLower.includes("exabot")) return "exabot";
  if (uaLower.includes("amazon") || uaLower.includes("amzn")) return "amazonbot";
  if (uaLower.includes("applebot")) return "applebot";
  if (uaLower.includes("twitterbot")) return "twitterbot";
  if (uaLower.includes("linkedinbot")) return "linkedinbot";
  if (uaLower.includes("facebook") || uaLower.includes("facebot")) return "facebookbot";
  if (uaLower.includes("pinterest")) return "pinterest";
  if (uaLower.includes("slackbot")) return "slackbot";
  if (uaLower.includes("discordbot")) return "discordbot";
  if (uaLower.includes("telegrambot")) return "telegrambot";
  if (uaLower.includes("whatsapp")) return "whatsapp";
  if (uaLower.includes("ia_archiver")) return "alexa";
  if (uaLower.includes("headlesschrome")) return "headless_chrome";
  if (uaLower.includes("phantomjs")) return "phantomjs";
  if (uaLower.includes("selenium")) return "selenium";
  if (uaLower.includes("puppeteer")) return "puppeteer";
  if (uaLower.includes("playwright")) return "playwright";
  if (uaLower.includes("curl")) return "curl";
  if (uaLower.includes("wget")) return "wget";
  if (uaLower.includes("python-requests")) return "python_requests";
  if (uaLower.includes("axios")) return "axios";
  if (uaLower.includes("node-fetch")) return "node_fetch";
  if (uaLower === "node" || uaLower.startsWith("node/")) return "node_client";
  if (uaLower.includes("go-http-client")) return "go_http";
  if (uaLower.includes("java/")) return "java_client";
  if (uaLower.includes("scrapy")) return "scrapy";
  if (uaLower.includes("nutch")) return "nutch";
  if (uaLower.includes("httpie")) return "httpie";
  if (uaLower.includes("libwww")) return "libwww";
  if (uaLower.includes("archive.org")) return "archive_org";
  if (uaLower.includes("embedly")) return "embedly";
  if (uaLower.includes("outbrain")) return "outbrain";
  const botMatch = uaLower.match(/([a-z0-9_-]+)(?:bot|crawler|spider)/);
  if (botMatch && botMatch[1] && botMatch[1].length > 1) {
    return `${botMatch[1]}_bot`;
  }
  if (uaLower.includes("bot") || uaLower.includes("crawl") || uaLower.includes("spider")) {
    return "unknown_bot";
  }
  return null;
}

function shouldSkipPath(pathname) {
  if (SKIP_PATH_PREFIXES.some((p) => pathname.startsWith(p))) return true;
  if (SKIP_PATH_EXACT.includes(pathname)) return true;
  if (SKIP_EXTENSIONS.some((ext) => pathname.endsWith(ext))) return true;
  return false;
}

function isProbePath(pathname) {
  return PROBE_PATH_PATTERNS.some((re) => re.test(pathname));
}

async function hashIP(ip) {
  const data = new TextEncoder().encode(ip + HASH_SALT);
  const hash = await crypto.subtle.digest("SHA-256", data);
  const arr = new Uint8Array(hash);
  return Array.from(arr.slice(0, 8))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

export default {
  async fetch(request, env, ctx) {
    const response = await fetch(request);

    // 统计只关心 GET 的页面类请求
    if (request.method !== "GET") return response;

    const url = new URL(request.url);
    const path = url.pathname;
    const ua = request.headers.get("user-agent") || "";

    const isFeedClient = FEED_UA_PATTERNS.some((re) => re.test(ua));
    const isFeedEndpoint = path === "/rss.xml" || path === "/feed.xml";

    // 静态资源两个数据集都不记，保证「真人 vs 爬虫」可直接对比。
    // 例外：RSS 阅读器取 /rss.xml —— 那是真实订阅行为，必须记下来，
    //      否则又会回到「订阅者被当成爬虫」或者干脆统计不到。
    if (!(isFeedClient || isFeedEndpoint) && shouldSkipPath(path)) return response;

    // 响应已经拿到了，剩下的统计工作放到响应之后做，不给访客增加延迟
    ctx.waitUntil(
      (async () => {
        try {
          await record(request, env, response, url);
        } catch (e) {
          // 原来这里是空 catch，数据写失败会无声丢失
          console.error(
            JSON.stringify({
              msg: "analytics write failed",
              path,
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
  const path = url.pathname;
  const hostname = url.hostname;
  const ua = request.headers.get("user-agent") || "";
  const ip = request.headers.get("cf-connecting-ip") || "unknown";
  const ipHash = await hashIP(ip);
  const status = response.status;
  const country = request.cf?.country || "";
  const city = request.cf?.city || "";
  const index = [hostname];

  const writeBot = (kind, botType) =>
    env.BOT_ANALYTICS.writeDataPoint({
      // blob2 原来是明文 IP，保留空位不写，避免破坏已有查询的列序
      blobs: [path, "", ua, hostname, country, ipHash, botType || kind, kind, String(status)],
      doubles: [1],
      indexes: index,
    });

  const writeHuman = (clientClass) =>
    env.ANALYTICS.writeDataPoint({
      // blob3 原来是明文 IP，保留空位不写
      blobs: [
        path,
        ipHash,
        "",
        request.headers.get("referer") || "",
        country,
        city,
        hostname,
        ua,
        clientClass,
        String(status),
      ],
      doubles: [1],
      indexes: index,
    });

  // ① 404 —— 不是页面浏览。这是旧版最大的污染源：
  //    扫描器打 /.env、/wp-login.php 都返回 404，但旧版只看 UA 就当成真人了。
  if (status === 404) {
    return writeBot("notfound", "notfound");
  }

  // ② RSS / Atom 阅读器 —— 真实订阅者，归入真人数据集但单独标记
  if (FEED_UA_PATTERNS.some((re) => re.test(ua))) {
    return writeHuman("feed");
  }

  // ③ 安全扫描器 / 资产测绘
  if (SCANNER_UA_PATTERNS.some((re) => re.test(ua))) {
    return writeBot("scanner", "scanner_ua");
  }
  if (isProbePath(path)) {
    return writeBot("scanner", "probe_path");
  }

  // ④ 空 UA / 过短 UA
  if (!ua || ua.length < 10) {
    return writeBot("empty_ua", "empty_ua");
  }

  // ⑤ 已知爬虫
  if (BOT_UA_PATTERNS.some((re) => re.test(ua))) {
    return writeBot("bot", classifyBot(ua) || "unknown_bot");
  }
  if (BOT_IP_PREFIXES.some((prefix) => ip.startsWith(prefix))) {
    return writeBot("bot", classifyBot(ua) || "bot_ip");
  }
  // 免费套餐下 request.cf.botManagement 不会注入，这段无副作用；
  // 将来升级到付费套餐会自动生效。
  const botScore = request.cf?.botManagement?.score;
  if (botScore !== undefined && botScore < 30) {
    return writeBot("bot", "bot_score");
  }

  // ⑥ 真人
  return writeHuman("browser");
}
