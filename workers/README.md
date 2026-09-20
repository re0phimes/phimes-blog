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

三个 Worker 的源码都在本目录下：
`workers/site-analytics/`、`workers/analytics-dashboard/`、`workers/blog-page-views/`。

> ⚠️ **Worker 不会自动部署**。`.github/workflows/deploy-pages.yml` 只部署 Pages 静态站，
> 不碰任何 Worker。改完 Worker 代码要手动部署一次（见下），否则线上还是旧的。

## blog-page-views

文章阅读量计数，`views.phimes.top`，用 Workers KV 存储（绑定名 `PAGE_VIEWS`）。

```
GET  /all?          → { "<path>": count, ... }   构建期 fetch-popular 用
GET  /?path=<路径>  → { path, count }
POST /?path=<路径>  → { path, count }            加一并返回最新值
```

⚠️ **不要用 `*.workers.dev`**：那个域名在国内被 DNS 污染（四个解析器返回四个不同的
随机社交网站 IP），浏览器请求根本到不了 Worker，表现就是文章页的火焰数字恒为 0。
必须绑自有域名，这也是 `views.phimes.top` 存在的原因。

⚠️ **POST 是无条件加一**，没有去重 —— 所以这是 PV 不是 UV。同一个人刷新一次就加一次。
真人/爬虫的区分在 `site-analytics` 那边做。

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

### 下钻（明细）

看板里所有表格的行都**可点击**（行首有个 `›`），点进去看该分组的原始记录：
时间 / 路径 / 国家 / 城市 / **IP 或哈希** / AS 组织 / Referer / 状态码 / 完整 UA。

**明细支持分页**：每页可选 50 / 100 / 200 条，页码窗口是当前页 ±2，
翻页会回到顶部并保持当前分组。

后端是 `/api/detail?range=&src=bot|human&dim=&val=&page=&pageSize=`，
`dim` 走白名单（见 `DETAIL_DIMS`），值经 `sqlStr()` 转义后拼进 SQL ——
Analytics Engine 的 SQL API 不支持参数化查询，所以这是必须的。
分页用 `LIMIT n OFFSET m`（实测 offset 20000 也正常），同时跑一条 `count()`
拿总数，两个查询并发。

**能看到的 IP 分两种：**

| 数据            | 显示                      | 说明                          |
| --------------- | ------------------------- | ----------------------------- |
| 2026-09-20 之前 | **明文 IP**               | 旧格式，共 26,296 + 40,728 条 |
| 2026-09-20 之后 | 加盐哈希（16 位十六进制） | 不能反推 IP                   |

明细页会明确标出「其中 N 条是旧数据，带明文 IP」。

### 改这个文件时的注意事项

看板的 HTML 和 JS 是嵌在 Worker 源码的**模板字符串**里的，
所以用脚本改它很容易踩坑 —— 我这次就踩了：

- 用「按标记切片再拼接」的方式改，**切掉的区间里可能还藏着别的函数**。
  我切 `drill()` 那次把紧随其后的 `chart()` 一起删了，
  结果页面只剩一句「加载失败：chart is not defined」。
- 改完**一定要做一次全量自检**：列出页面脚本里调用的所有函数名，
  逐个确认有定义。只检查「没报语法错」是不够的。

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
