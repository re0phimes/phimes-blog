# Cloudflare Workers

这两个 Worker 不在构建流程里，改动后需要手动部署。

## 当前线上路由

| 路由                 | Worker                | 作用                      |
| -------------------- | --------------------- | ------------------------- |
| `phimes.top/*`       | `site-analytics`      | 统计入口（见下）          |
| `*.phimes.top/*`     | `site-analytics`      | 同上，兜底所有子域        |
| `views.phimes.top/*` | `blog-page-views`     | 文章阅读量 API（KV 存储） |
| `stats.phimes.top/*` | `analytics-dashboard` | 统计看板（Basic Auth）    |

路由按**具体程度**匹配，所以 `views` 和 `stats` 会覆盖 `*.phimes.top/*` 那条兜底。

## site-analytics

把访问分流写进两个 Analytics Engine 数据集：

- `site_visits` —— 真实读者（blob9 = `browser` | `feed`）
- `bot_visits` —— 爬虫 / 扫描器 / 空 UA / 404 探测（blob8 = kind）

blob 布局：

```
site_visits: 1 path | 2 ipHash | 3 ""(保留) | 4 referer | 5 country
             6 city | 7 hostname | 8 ua | 9 clientClass | 10 status
bot_visits:  1 path | 2 ""(保留) | 3 ua | 4 hostname | 5 country
             6 ipHash | 7 botType | 8 kind | 9 status
```

> blob3/blob2 原来是**明文 IP**，2026-09-20 起不再写入，位置保留以兼容旧查询。

### 部署（注意两点）

```bash
# 1) 模块文件名必须和 metadata 里的 main_module 完全一致，
#    否则报 "No such module"（Cloudflare 用的是 multipart 的文件名）
# 2) bindings 必须一起上传，否则会把 ANALYTICS / BOT_ANALYTICS 清空
curl -X PUT "https://api.cloudflare.com/client/v4/accounts/$ACC/workers/scripts/site-analytics" \
  -H "Authorization: Bearer $CF_TOKEN" \
  -F 'metadata=@metadata.json;type=application/json' \
  -F 'site-analytics.js=@workers/site-analytics/index.js;type=application/javascript+module'
```

`metadata.json`：

```json
{
  "main_module": "site-analytics.js",
  "compatibility_date": "2024-09-23",
  "bindings": [
    { "type": "analytics_engine", "name": "ANALYTICS", "dataset": "site_visits" },
    { "type": "analytics_engine", "name": "BOT_ANALYTICS", "dataset": "bot_visits" }
  ]
}
```

## analytics-dashboard

只读看板，Basic Auth 保护（密码在 Worker secret `DASH_PASSWORD`）。
另外需要两个 secret：`CF_ANALYTICS_TOKEN`（权限只要 Account Analytics Read）、`CF_ACCOUNT_ID`。

功能：时间范围切换（1h / 6h / 24h / 7d / 30d / 90d，≤24h 按小时聚合，其余按天）、
真人/爬虫趋势对比、爬虫分类、访问最多页面、来源国家、设备系统、Referer、
以及「内容优化建议」（对比每篇文章的真人阅读量 vs 爬虫抓取次数）。

### 判定逻辑（v3，2026-09-20）

顺序很重要，第一个命中决定归类：

| #   | 条件                                                                        | 结果                    |
| --- | --------------------------------------------------------------------------- | ----------------------- |
| ①   | 路径不在正向白名单（`/`、`/posts/`、`/pages/`、`/page`、feed/crawler 端点） | `notfound`              |
| ②   | 状态码不是 200 / 304                                                        | `http_<status>`         |
| ③   | UA 命中扫描器特征                                                           | `scanner`               |
| ④   | 路径命中探针特征（`/.env`、`/wp-*`…）                                       | `scanner`               |
| ⑤   | UA 为空或过短                                                               | `empty_ua`              |
| ⑥   | UA 命中 RSS 阅读器白名单                                                    | `feed`（**真人**）      |
| ⑦   | UA 命中搜索引擎特征                                                         | `search_bot`            |
| ⑧   | UA 命中 AI 爬虫特征                                                         | `ai_bot`                |
| ⑨   | UA 命中其他爬虫/命令行工具                                                  | `bot`                   |
| ⑩   | IP 命中已知爬虫网段                                                         | `bot`                   |
| ⑪   | **AS 组织名是机房**（DigitalOcean / AWS / OVH…）                            | `datacenter`            |
| ⑫   | `cf.botManagement.score < 30`                                               | `bot`（免费套餐不触发） |
| ⑬   | 以上都不匹配                                                                | `browser`（**真人**）   |

**为什么 ① 用正向白名单而不是状态码**：原来只排除 404，但浏览器缓存过
一个 404 页面后再访问，Cloudflare 会回 `304 Not Modified`，于是绕过了判定
被算成真人（实测 `/NCWHtoBSD.html`、`/embedding_visulization.html` 就是这样
混进来的）。改成正向白名单后与状态码无关。

**为什么 ⑥ 的白名单要收窄**：原来有 `/rss/i` 和 `/feed/i` 这种宽泛匹配，
`python-feedparser`（抓取库）名字里带 feed，会被误判成真实订阅者。

**⑪ 为什么不用 `botManagement`**：那个字段需要付费套餐才会注入，免费版是
`undefined`，等于死代码。`asOrganization` 免费版就能读到。

### 一个诚实的说明：IP 哈希的强度

`ipHash = SHA-256(IP + salt)` 取前 8 字节，用于 UV 去重。

盐通过 Worker secret `IP_HASH_SALT` 注入，源码里不再出现。**但这个值
在仓库的 git 历史里出现过**，所以：

- 想让它真正不可反推（IPv4 只有 2³² 个地址，暴力反查是可行的），需要
  **换一个新盐**；
- 代价是**新旧哈希不一致，UV 会有一段重叠期**（同一人被算两次），
  直到 3 个月保留期把旧数据滚掉。

现在用的是旧值（保证 UV 连续），换不换由你决定。

### 两个 Analytics Engine SQL 的坑

AE 的 SQL 是 ClickHouse 的**受限子集**，实测：

| 写法                       | 结果                                |
| -------------------------- | ----------------------------------- |
| `multiIf(...)`             | ❌ `unknown function call: MULTIIF` |
| `CASE WHEN ... END`        | ❌ `unsupported expression type`    |
| `if(a,b,c)`（可嵌套）      | ✅                                  |
| `toStartOfHour(timestamp)` | ✅                                  |
| `count(DISTINCT blob)`     | ✅                                  |
| `uniq()`                   | ❌ `unknown function call: UNIQ`    |

### 一个模板字符串的坑

看板的内联 `<script>` 是嵌在 Worker 源码的**模板字符串**里的。
模板字符串会吃掉一层反斜杠：写 `\/` 到页面就变成 `/`，
于是 `p.replace(/^\/posts\//, '')` 到浏览器变成 `/^/posts//` —— **非法正则**，
整个脚本直接 SyntaxError，页面只剩一个「加载中…」。

所以那段代码里**不要用带反斜杠转义的正则**，用 `indexOf` / `slice` 代替。
ESLint 的 `no-useless-escape` 也能顺带发现这个问题。

## 常用查询

Analytics Engine SQL API：

```bash
curl -X POST "https://api.cloudflare.com/client/v4/accounts/$ACC/analytics_engine/sql" \
  -H "Authorization: Bearer $CF_ANALYTICS_TOKEN" -H "Content-Type: text/plain" \
  --data "SELECT blob1 AS path, count() AS n FROM site_visits
          WHERE timestamp > NOW() - INTERVAL '30' DAY AND blob9 != 'feed'
          GROUP BY path ORDER BY n DESC LIMIT 20"
```

| 想查什么     | 关键条件                            |
| ------------ | ----------------------------------- |
| 真人页面浏览 | `site_visits` + `blob9 != 'feed'`   |
| RSS 订阅者   | `site_visits` + `blob9 = 'feed'`    |
| 爬虫总数     | `bot_visits`                        |
| 只在扫描器   | `bot_visits` + `blob8 = 'scanner'`  |
| 404 探测     | `bot_visits` + `blob8 = 'notfound'` |
| 按爬虫类型   | `GROUP BY blob7`（botType）         |
| 真人 UV      | `count(DISTINCT blob2)`（IP 哈希）  |

> ⚠️ `blob8` / `blob9` 是 2026-09-20 才加的。之前的旧数据没有分类，
> 所以历史「真人」统计里仍含扫描器噪音。
