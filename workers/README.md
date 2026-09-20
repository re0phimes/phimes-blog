# Cloudflare Workers

这两个 Worker 不在构建流程里，改动后需要手动部署。

## 当前线上路由

| 路由 | Worker | 作用 |
|---|---|---|
| `phimes.top/*` | `site-analytics` | 统计入口（见下） |
| `*.phimes.top/*` | `site-analytics` | 同上，兜底所有子域 |
| `views.phimes.top/*` | `blog-page-views` | 文章阅读量 API（KV 存储） |
| `stats.phimes.top/*` | `analytics-dashboard` | 统计看板（Basic Auth） |

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

## 常用查询

Analytics Engine SQL API：

```bash
curl -X POST "https://api.cloudflare.com/client/v4/accounts/$ACC/analytics_engine/sql" \
  -H "Authorization: Bearer $CF_ANALYTICS_TOKEN" -H "Content-Type: text/plain" \
  --data "SELECT blob1 AS path, count() AS n FROM site_visits
          WHERE timestamp > NOW() - INTERVAL '30' DAY AND blob9 != 'feed'
          GROUP BY path ORDER BY n DESC LIMIT 20"
```

| 想查什么 | 关键条件 |
|---|---|
| 真人页面浏览 | `site_visits` + `blob9 != 'feed'` |
| RSS 订阅者 | `site_visits` + `blob9 = 'feed'` |
| 爬虫总数 | `bot_visits` |
| 只在扫描器 | `bot_visits` + `blob8 = 'scanner'` |
| 404 探测 | `bot_visits` + `blob8 = 'notfound'` |
| 按爬虫类型 | `GROUP BY blob7`（botType） |
| 真人 UV | `count(DISTINCT blob2)`（IP 哈希） |

> ⚠️ `blob8` / `blob9` 是 2026-09-20 才加的。之前的旧数据没有分类，
> 所以历史「真人」统计里仍含扫描器噪音。
