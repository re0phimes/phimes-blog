# 项目现状与交接文档

> 最后更新：2026-09-20 · 对应提交 `29c09bd`
> 这份文档是给「新开一个对话继续改这个站」用的，读完应该能直接上手。

---

## 1. 这是什么

**Phimes 的个人技术博客** —— 一个中文技术博客，内容集中在大模型原理与推理优化。

|          |                                                     |
| -------- | --------------------------------------------------- |
| 线上地址 | https://blog.phimes.top/                            |
| 仓库     | https://github.com/re0phimes/phimes-blog （public） |
| 文章数   | 22 篇                                               |
| 主题代码 | 约 13,900 行                                        |
| 统计看板 | https://stats.phimes.top/ （Basic Auth）            |
| 分支     | `master`（本地 = 远端）                             |

---

## 2. 整体架构

```
用户
 │
 ├─ blog.phimes.top ──→ Cloudflare（DNS + 代理）
 │                       └─ Cloudflare Pages（静态文件就在 CF 边缘，没有"回源"这一跳）
 │
 ├─ image.phimes.top ─→ Cloudflare ──→ 阿里云 OSS 香港（图片，已 WebP 化）
 │
 ├─ views.phimes.top ─→ Worker: blog-page-views   文章阅读量（KV 存储）
 ├─ stats.phimes.top ─→ Worker: analytics-dashboard  统计看板
 │                      （源码都在仓库 workers/ 下，三个 Worker 都需要手动部署）
 │
 └─ phimes.top / www / aifaq / demo / quiz ─→ Vercel（**尚未迁移**）
```

**统计数据流**（所有 `*.phimes.top` 流量都会先经过这个 Worker）：

```
请求 → Worker: site-analytics → 转发给源站 → 返回给访客
                └─ ctx.waitUntil() 异步分类并写入 Analytics Engine
                     ├─ site_visits  真人（browser / feed）
                     └─ bot_visits   爬虫（search_bot / ai_bot / scanner / datacenter / notfound …）
```

Worker 路由优先级（更具体的覆盖通配）：

```
phimes.top/*          → site-analytics
*.phimes.top/*        → site-analytics        ← 兜底
views.phimes.top/*    → blog-page-views       ← 覆盖兜底
stats.phimes.top/*    → analytics-dashboard   ← 覆盖兜底
```

---

## 3. 技术栈

| 层       | 用什么                                                                                   |
| -------- | ---------------------------------------------------------------------------------------- |
| 站点生成 | VitePress 1.6.4 + Vue 3                                                                  |
| 主题     | fork 自 `vitepress-theme-blog`，已大幅改造                                               |
| 样式     | SCSS（`main.scss` 变量 + `post.scss` 正文）                                              |
| 组件     | `unplugin-vue-components` 按目录**自动导入**（模板里直接写 `<PostList />`，不用 import） |
| 全局 API | `unplugin-auto-import`（`ref` / `computed` / `useData` 等不用 import）                   |
| 数学公式 | 自定义 `markdownKatex.mjs`（KaTeX 0.16）                                                 |
| 部署     | GitHub Actions 构建 → `wrangler pages deploy` 直传 Cloudflare Pages                      |
| 图片     | 阿里云 OSS 香港 + WebP；封面构建期下载并转 WebP                                          |
| 统计     | Cloudflare Workers + Analytics Engine                                                    |

---

## 4. 功能清单

### 4.1 视觉 / 排版

- **亮色主题**：Anthropic 配色（珊瑚 `#cc785c` + 暖米白 `#faf9f5`）
- **暗色主题**：Anthropic 暖色暗面（`#181715` / `#1f1e1b` / `#252320`）
- **字体**：系统字体栈（`-apple-system` + PingFang SC 等），**零 webfont 下载**
  - 中英文由操作系统成对提供，天然匹配
  - 等宽字体也是系统栈（代码块）
- **正文加粗**用珊瑚色高亮（`--main-highlight-color`）
- **图片**：全部 WebP、带 `width`/`height`（CLS = 0）、`decoding="async"`
- **代码块**、KaTeX 公式、表格横向滚动都已处理

### 4.2 首页

- **Hero 头条**：最新文章，珊瑚标签 + 大标题 + 手写摘要 + 「阅读全文 →」
- **三栏封面次条**：最近 3 篇的封面卡
- **历史文章区**：
  - 「时间 / 热度 / 标签」三段筛选，1px 分割线
  - 时间 = 最新优先；热度 = 按浏览量；标签 = 下拉浮层
  - 客户端分页，状态写进 URL（可分享）
  - 翻页控件与页脚对齐

### 4.3 文章页

- 目录（TOC，滚动高亮）
- **火焰图标 + 阅读量**（来自 `views.phimes.top`）
- 热门标签 / 相关标签（标签页时显示共现标签）
- 站点数据（文章总数）
- 版权、上下篇、图片灯箱（点击放大）
- 支持 `[[笔记名]]` Obsidian 双链语法

### 4.4 其他页面

`/pages/about`（Phimes 自己的介绍）、`/pages/project`（四个站点卡片）、
`/pages/link`（友链申请，列表待自己填）、`/pages/archives`、`/pages/tags`、`/pages/categories`、
`/pages/privacy`、`/pages/cc`

### 4.5 SEO / AI Agent 友好

| 项            | 状态                                                                                                                                    |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `robots.txt`  | ✅ 显式放行 AI 爬虫（GPTBot / ClaudeBot / PerplexityBot / CCBot / Bytespider / DeepSeekBot / GrokBot …）与 RSS 客户端；屏蔽扫描探针路径 |
| `llms.txt`    | ✅ 构建期自动生成（`generateLlmsTxt.mjs`），22 篇 + 全部标签                                                                            |
| JSON-LD       | ✅ 全站 64 页；文章页有完整 `BlogPosting`（日期/标签/封面/作者）                                                                        |
| `sitemap.xml` | ✅ 58 URL，**58/58 有 lastmod**，21 篇带 `image:image`；已排除 `/page/N` 空壳                                                           |
| RSS           | ✅ `/rss.xml`                                                                                                                           |
| 文章索引      | ✅ `/index.json`                                                                                                                        |
| 404           | ✅ 真 404（不是软 404）                                                                                                                 |

### 4.6 统计看板（stats.phimes.top）

- **时间维度**：1 小时 / 6 小时 / 24 小时 / 7 天 / 30 天 / 90 天（≤24h 按小时聚合，其余按天）
- **真人 vs 爬虫**趋势双色对比图
- **爬虫分类**：`search_bot`（搜索引擎）/ `ai_bot`（GPTBot、ClaudeBot、Perplexity…）/ `scanner` / `datacenter` / `notfound` / `empty_ua` / `http_*` / `bot`
- **真人维度**：Top 页面、来源国家、设备系统、Referer
- **内容优化建议**：自动对比每篇文章的「真人阅读量」vs「爬虫抓取次数」，
  分出「被抓很多没人读」（→ 改标题摘要）和「有人读没被抓」（→ 补内链）
- **下钻**：任意表格的行可点击，展开原始记录（时间/路径/国家/城市/**IP 或哈希**/AS 组织/Referer/状态码/完整 UA）
- **下钻分页**：每页 50 / 100 / 200

### 4.7 安全 / 基础设施

| 项                           | 状态                                                       |
| ---------------------------- | ---------------------------------------------------------- |
| **DNSSEC**                   | ✅ 已启用并生效（key_tag `2371`，DS 记录在阿里云注册商侧） |
| **HSTS**                     | ✅ `max-age=15552000; includeSubDomains`                   |
| 最低 TLS                     | 1.2                                                        |
| TLS 1.3 + 0-RTT              | ✅                                                         |
| Brotli / HTTP3 / Early Hints | ✅ 全开                                                    |
| SSL 模式                     | Full (strict)                                              |
| 图片防盗链                   | ✅ OSS Referer 白名单（含 localhost），且已收紧通配符      |
| 假备案号                     | ✅ 已移除                                                  |

---

## 5. 本次会话做了什么（22 个提交，五个阶段）

### 阶段一：图片防盗链修复（`235deb9`）

`blog.phimes.top` 的图片在本地 dev 全部 403。根因是 OSS 防盗链白名单里的
`127.0.0.1` / `localhost` **缺 scheme**，匹配不上浏览器实际发的
`http://127.0.0.1:9877/`。加上 `http://localhost:*` / `http://127.0.0.1:*`，
并把 `https://*.phimes.top` 收紧成 `https://*.phimes.top/*`
（原来的宽松匹配让 `evil.com/blog.phimes.top` 都能过）。

### 阶段二：全盘 system debug（`40f29d0`）

扫出并修掉 18 个问题，其中：

- **About / Project / Link 三个页面用的是上游作者已死的图床** `pic.efefee.cn`
  （DNS 已不存在），而且 `pages/about` 整页写的是「我是**無名**」，
  `pages/link.md` 在教访客添加「無名小栈」的友链
- **图片灯箱全站失效**（Fancybox 依赖的第三方 CDN 取不到）
- `/pages/about` 抛 `TypeError`（51.la 接口已 404，代码没做空值保护）
- **`Countdown` 是死组件**：模板整段注释掉、渲染为空，但每页产生一条 Vue 警告，
  还在跑 1 秒的 `setInterval`，而且挂载时没有 `v-if`
- **`npm run lint` 从来没跑起来过**（`.eslintrc.js` 是 CommonJS 但 package 是 ESM）
- **`fetch-popular` 超时会静默清空缓存**
- `.vitepress/init.mjs` 用了 ESM 里不存在的 `__dirname`

### 阶段三：深色主题 + 图片优化（`8fc8e87`）

- 暗色换成 Anthropic 暖色暗面；顺手修掉**语义色从 `:root` 继承亮色值**的问题
  （暗背景上看不清）
- **正文图片 19.2 MB → 6.9 MB**（WebP，136 张）
- **封面 4.2 MB → 1.3 MB**
- 删掉 **7 MB 的死壁纸**（`Background.vue` 从来没渲染过它）
- 用图片尺寸清单消除 CLS

### 阶段四：字体 + 性能 + 迁移（`809dae9` → `f2f9f89`）

- **字体是最大的性能问题**：HarmonyOS Sans 被拆成 46 个子集文件共 723 KB（占全站 55%），
  而且它的拉丁和 CJK 是两套设计。换成**系统字体栈**，**零下载**
- 移除 Microsoft Clarity（会话录制，隐私成本最高）
- **Vercel → Cloudflare Pages 迁移**：
  - 域名 CNAME 从 Vercel 改到 `phimes-blog.pages.dev`
  - 建立 GitHub Actions 直传部署链路
  - `vercel.json` → `public/_headers`
  - 顺带修掉软 404
- **修复「热度」恒为 0**：根因是 **`*.workers.dev` 在国内被 DNS 污染**，
  给 Worker 绑了自有域名 `views.phimes.top`

### 阶段五：SEO + 统计（`17ac8bc` → `29c09bd`）

- **修掉 `robots.txt` 的 Sitemap 指向上游作者域名** `blog.imsyy.top`
- 新增 `llms.txt`、JSON-LD、sitemap 优化
- **重写真人/爬虫判定逻辑（v3）**，修掉 4 个错判：
  1. **304 绕过了 notfound 判定** —— 缓存过的 404 页面再访问会返回 304，
     被算成真人（`.env`、`/wp-login.php` 这些就是这样混进来的）
  2. **RSS 白名单太宽** —— `/feed/i` 会匹配 `python-feedparser` 这种抓取库
  3. 探针路径依赖状态码
  4. 非 2xx 全算真人
- 新增能力：`asOrganization` 机房识别、搜索引擎与 AI 爬虫分开、爬虫端点记录
- 看板：时间维度、多维度分析、内容优化建议、**下钻 + 分页**
- 两个 Worker 源码纳入版本管理（`workers/`）

---

## 6. 已知陷阱（踩过的坑，务必先看）

### 6.1 `themeConfig` 是**浅合并**

`themeConfig.mjs`（用户配置）通过 `Object.assign` 覆盖
`.vitepress/theme/assets/themeConfig.mjs`（默认配置）。

> ⚠️ **任何嵌套对象只要在用户配置里写了一半，就会整个替换掉默认值。**
> 例如写 `aside: { tags: {...} }`，那 `aside.toc`、`aside.siteData` 就都没了。

### 6.2 Analytics Engine SQL 是 **ClickHouse 受限子集**

| 写法                               | 结果                                |
| ---------------------------------- | ----------------------------------- |
| `multiIf(...)`                     | ❌ `unknown function call: MULTIIF` |
| `CASE WHEN ... END`                | ❌ `unsupported expression type`    |
| `if(a,b,c)`（可嵌套）              | ✅                                  |
| `uniq()`                           | ❌ `unknown function call: UNIQ`    |
| `count(DISTINCT x)`                | ✅                                  |
| `toStartOfHour(ts)` / `toDate(ts)` | ✅                                  |
| `LIMIT n OFFSET m`                 | ✅（offset 20000 也正常）           |

**没有删除接口**：写入的数据 3 个月滚动过期，期间删不掉。
目前 `site_visits` 有 26,296 条、`bot_visits` 有 40,728 条**旧数据仍带明文 IP**。

### 6.3 Worker 源码里的**模板字符串会吃掉反斜杠**

看板的 HTML/JS 嵌在 Worker 源码的模板字符串里。写 `\/` 到浏览器会变成 `/`，
于是 `p.replace(/^\/posts\//,'')` 变成非法正则 `/^/posts//`，
**整个脚本 SyntaxError，页面只剩一个「加载中…」**，而且 Worker 侧毫无报错。

**结论：那段代码里不要用带反斜杠转义的正则，用 `indexOf` / `slice` 代替。**

### 6.4 改模板字符串里的代码必须做**全量自检**

用「按标记切片再拼接」的方式改，**切掉的区间里可能藏着别的函数**。
我切 `drill()` 时把紧随其后的 `chart()` 一起删了，页面只剩
「加载失败：chart is not defined」，且没有任何语法报错。

**改完一定要列出页面脚本调用的所有函数名，逐个确认有定义。**

### 6.5 部署 Worker 的两个坑

1. **上传的 multipart 文件名必须和 `metadata.main_module` 完全一致**，
   否则报 `No such module`（Cloudflare 用的是文件名，不是字段名）
2. **bindings 必须一起上传**，否则会把 Analytics Engine 绑定清空

ES Module 的 Content-Type 要用 `application/javascript+module`。

### 6.6 GitHub Actions：`secrets` 不能用在 step 级 `if`

写了会被**校验阶段直接拒绝**，表现是 **0 秒失败且没有任何 job 日志**。
只能先把 secret 读进 `env`，再用 step output 控制。

### 6.7 VitePress 的 `useRoute()` **不是 vue-router**

它是自定义的精简路由，`route` 上**没有 `query` / `fullPath`**。
要读 query 得用 `window.location`，改 query 用 `history.replaceState`。

### 6.8 本机环境问题（只影响本地开发，不影响线上）

- **本地代理 `127.0.0.1:7897` 不稳定**：对 `github.com`、`image.phimes.top`、
  `*.workers.dev` 间歇性 TLS 失败（`SSL_ERROR_SYSCALL`）。
  绕过用 `--noproxy '*'`，git 用 `env -u http_proxy -u https_proxy -u all_proxy git push`
- **SSH 可用来推 git**：`git push git@github.com:re0phimes/phimes-blog.git master`
  （`gh` 的 OAuth token 没有 `workflow` scope，推 `.github/workflows/` 下的改动会被拒）
- **npm 缓存权限问题**：`~/.npm/_cacache` 有 root 文件，需要
  `npm_config_cache=/tmp/dsh-npm-cache`
- 本机 DNS 有时只缓存到 AAAA（IPv6 不通）→ `ERR_ADDRESS_UNREACHABLE`

---

## 7. 待办事项

### P0 · 安全（越早越好）

- [ ] **轮换阿里云 AK/SK** —— 这把密钥已在会话中出现多次，应当视为已泄漏
      （RAM 控制台禁用重建；新建时只授予实际需要的权限，例如
      `AliyunDomainFullAccess` 或更细的 `domain:DnsSecSetting`）
- [ ] **轮换 Cloudflare 大权限 token** —— 同理
- [ ] **改 `stats.phimes.top` 的登录密码**（见 8.3）

### P1

- [ ] **换 IP 哈希盐** —— 现在的盐在 git 历史里（仓库 public），
      IPv4 只有 2³² 个地址所以理论上可暴力反查。换新盐的代价是
      **UV 会有一段重叠期**（同一人被算两次），直到 3 个月后旧数据滚掉。
      操作：更新 Worker secret `IP_HASH_SALT`
- [ ] **字体子集** —— 如果想换回楷体（霞鹜文楷），
      建议自建子集：全站只用了 **1536 个不同汉字**，
      子集后约 300–500 KB（对比官方 CDN 的 425 KB CSS + 1.5–2.5 MB 字体），
      而且能放到自己的 OSS 上，不再依赖那个不稳定的 CDN
- [ ] **删除 Vercel 上的 blog 项目 + 孤儿 TXT 记录**
      （`_vercel.phimes.top` 里 `vc-domain-verify=blog.phimes.top,…` 那条已失效）

### P2

- [ ] **统计长期归档** —— Analytics Engine 只保留 3 个月且无法导出。
      要长期保存得加定时导出（Worker Cron Trigger 或 GitHub Actions）
- [ ] **其余 6 个域名是否也搬到 Cloudflare Pages**
      （`phimes.top` / `www` / `aifaq` / `demo` / `quiz` / `preview.aifaq` 还在 Vercel，
      和图床迁移前的 blog 有完全一样的问题：`pdx1`/`sfo1` = 美西，每次访问跨境回源）
- [ ] **标签页瘦身** —— 23 个标签页里不少只有 1 篇文章（薄内容），
      可以考虑合并标签
- [ ] **sitemap 的 29 个索引页补 lastmod 之外的进一步优化**
- [ ] 看板加**区间对比**（本周 vs 上周）

### P3 · 可选

- [ ] Cloudflare Git 集成模式（需要浏览器授权 GitHub App；能换来**预览部署**和**面板一键回滚**，
      但要删掉现在的 Pages 项目重建）
- [ ] 图片进一步优化（AVIF）
- [ ] `themeConfig` 改成深合并，消除 6.1 的隐雷

---

## 8. 常用操作手册

### 8.1 本地开发

```bash
npm install
npm run dev          # http://localhost:9877
npm run build        # 构建前会先跑 fetch-popular
npm run lint
```

> ⚠️ **改 `themeConfig.mjs` 或文章 frontmatter 后需要重启 dev server**
> （`themeConfig` / `postData` 只在启动时构建一次）

### 8.2 部署

推送到 `master` → GitHub Actions 自动构建并直传 Cloudflare Pages（约 80 秒）。

- 手动部署：`npm run deploy`
- 触发条件只有推 `master` 和手动 `workflow_dispatch`；**推其他分支不会触发**
- 构建日志在 GitHub Actions 页面，不在 Cloudflare 面板

### 8.3 改看板密码

**方法 A（推荐）**：Cloudflare Dashboard → Workers & Pages → `analytics-dashboard`
→ Settings → Variables and Secrets → `DASH_PASSWORD` → Edit

**方法 B**：

```bash
curl -X PUT "https://api.cloudflare.com/client/v4/accounts/2cb7ae35f9305d1bfc18cf40172f6bdb/workers/scripts/analytics-dashboard/secrets" \
  -H "Authorization: Bearer <CF_TOKEN>" -H "Content-Type: application/json" \
  --data '{"name":"DASH_PASSWORD","text":"新密码","type":"secret_text"}'
```

### 8.4 部署 Worker

```bash
# 模块文件名必须与 metadata.main_module 一致；bindings 必须带上
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

### 8.5 查统计（Analytics Engine SQL）

```bash
curl -X POST "https://api.cloudflare.com/client/v4/accounts/$ACC/analytics_engine/sql" \
  -H "Authorization: Bearer $CF_ANALYTICS_TOKEN" -H "Content-Type: text/plain" \
  --data "SELECT blob1 AS path, count() AS n FROM site_visits
          WHERE timestamp > NOW() - INTERVAL '30' DAY AND blob9 != 'feed'
          GROUP BY path ORDER BY n DESC LIMIT 20"
```

**blob 布局：**

```
site_visits: 1 path | 2 ipHash | 3 ""(原明文IP,已停写) | 4 referer | 5 country
             6 city | 7 hostname | 8 ua | 9 clientClass | 10 status | 11 asOrganization
bot_visits:  1 path | 2 ""(原明文IP,已停写) | 3 ua | 4 hostname | 5 country
             6 ipHash | 7 botType | 8 kind | 9 status | 10 asOrganization
```

**常用条件：**
| 想查什么 | 条件 |
|---|---|
| 真人页面浏览 | `site_visits` + `blob9 != 'feed'` |
| RSS 订阅者 | `site_visits` + `blob9 = 'feed'` |
| 爬虫总数 | `bot_visits` |
| 只算扫描器 | `bot_visits` + `blob8 = 'scanner'` |
| 只算 AI 爬虫 | `bot_visits` + `blob8 = 'ai_bot'` |
| 机房流量 | `bot_visits` + `blob8 = 'datacenter'` |
| 真人 UV | `count(DISTINCT blob2)` |

> ⚠️ `blob8` / `blob9` / `blob10` / `blob11` 是 2026-09-20 才加的，
> **之前的旧数据没有分类**，历史「真人」统计里仍含扫描器噪音。

### 8.6 DNSSEC 相关

```
Cloudflare 侧：zone 1a8177e5b64800957a09256e2a0735ef，status = active，key_tag = 2371
阿里云侧：DS 记录已添加（产品 Domain / 2018-01-29）

删除 DS 记录：Action = SaveSingleTaskForDeletingDSRecord
查询：        Action = QueryDSRecord
验证：        https://dnsviz.net/
```

> ⚠️ **要把 DNS 从 Cloudflare 迁走，必须先删 DS 记录**，否则域名解析不了。

---

## 9. 账号与凭据（**值不写在这里**，仓库是 public）

| 项目                             | 位置                                                                                                            |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Cloudflare 账户 ID               | `2cb7ae35f9305d1bfc18cf40172f6bdb`                                                                              |
| Cloudflare zone ID（phimes.top） | `1a8177e5b64800957a09256e2a0735ef`                                                                              |
| Cloudflare Pages 项目            | `phimes-blog` → `phimes-blog.pages.dev`                                                                         |
| GitHub Secrets                   | `CLOUDFLARE_API_TOKEN`（Pages:Edit）、`CLOUDFLARE_ACCOUNT_ID`、`FAQ_DEPLOY_HOOK`                                |
| Worker secrets                   | `site-analytics`: `IP_HASH_SALT`；`analytics-dashboard`: `CF_ANALYTICS_TOKEN`、`CF_ACCOUNT_ID`、`DASH_PASSWORD` |
| 阿里云 OSS bucket                | `phimesimage`（`oss-cn-hongkong`），CNAME `image.phimes.top`                                                    |
| 域名注册商                       | 阿里云（万网）                                                                                                  |

---

## 10. 快速上手建议

拿到这个仓库后，先做这三件事确认状态：

```bash
# 1. 确认代码状态
git log --oneline -5
npm run lint && npx vitepress build

# 2. 确认线上正常
for p in / /sitemap.xml /llms.txt /nope; do
  curl -s -o /dev/null -w "%{http_code} $p\n" "https://blog.phimes.top$p"
done

# 3. 确认统计链路
curl -s -o /dev/null -w "views: %{http_code}\n" -X POST \
  "https://views.phimes.top?path=%2Ftest"
```

**改代码前先读第 6 节「已知陷阱」** —— 那里的每一条都是我实际踩过的，
尤其是 6.1（浅合并）、6.3/6.4（模板字符串与切片）、6.5（部署 Worker）。
