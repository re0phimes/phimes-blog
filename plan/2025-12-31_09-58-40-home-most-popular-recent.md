---
mode: plan
cwd: /home/modelenv/chentianxuan/s_projects/phimes-blog
task: 博客首页新增 Most Popular 与 Recent Posts 模块（数据来源与样式结构分析）
complexity: medium
planning_method: builtin
created_at: 2025-12-31T09:58:40+08:00
---

# Plan: 首页增加 Most Popular / Recent Posts 模块

🎯 任务概述
当前项目是基于 VitePress 的自定义主题（Vue 3 + SCSS）。首页由 `index.md` 指定 `layout: home`，实际渲染逻辑在 `.vitepress/theme/views/Home.vue`：左侧为文章列表（含分类/标签筛选与分页），右侧为 Aside 小组件。目标是在首页文章列表区域的开头新增一个“杂志式”模块，包含 `Most Popular` 与 `Recent Posts` 两个板块，并确认项目内是否已有可用的“热度/人气”数据记录。

📋 执行计划
1. **盘点现有数据能力（确定“Most Popular”可行性）**
   - 确认 `theme.postData` 的字段来源与排序规则（目前：`top` 置顶优先 + 时间倒序）。
   - 排查是否存在可复用的 PV/热度数据：当前文章页展示了 `artalk-pv-count`（运行时注入），但未进入 `postData`，也没有全站“按 PV 排序”的离线数据。
   - 结论落地为一条决策：Most Popular 采用 **手工策展**（推荐/置顶/配置列表），还是引入 **外部统计源**（Artalk/Twikoo/Clarity 等）。

2. **明确产品定义与交互边界（减少返工）**
   - 模块出现条件：仅在“真正首页”显示（非分类/标签页、非 `/page/2+` 分页页）。
   - 每个板块展示数量（建议 5–8），以及每条展示信息（标题 + 日期 + 可选分类/标签/摘要）。
   - 明确“Most Popular”命名是否可以接受“精选/推荐”的语义（如果没有真实 PV 数据）。

3. **选择 Most Popular 数据源（两条可落地路线，先选 A，B 为增强）**
   - A. **策展式（推荐优先，最快落地）**
     - 方案 A1：复用 frontmatter `top: true` 作为 Most Popular（语义偏“置顶/推荐”，不是“真实热门”）。
     - 方案 A2：新增 frontmatter 字段（更语义化）：`popular: true` 或 `popular_rank: 100`，并在构建期读取进 `postData`。
     - 方案 A3：在 `themeConfig.mjs` 增加显式列表（按文章路径/ID）：`home.mostPopular = ["/posts/...html", ...]`，完全可控。
   - B. **统计式（需要外部依赖，风险更高）**
     - 若已部署并启用 Artalk/Twikoo，调研其服务端是否可提供“按页面 PV 排序”的接口；如可用，在构建期拉取并缓存为 `popular.json`，再合并进 `themeConfig`。
     - 若依赖 Clarity/Vercel Analytics，需要额外 API Key 与数据导出流程，建议后置。

4. **实现数据选择器（Recent 与 Popular 的纯函数化）**
   - 在主题侧新增一个 selector（建议新文件）：`.vitepress/theme/utils/homeSections.mjs`（或在 `getPostData.mjs` 中补充导出函数）。
   - `recentPosts`：按 `date` 倒序取前 N（必要时可排除 `top` 文章，避免与 popular 重叠）。
   - `mostPopularPosts`：按“方案 A/B”得到的列表映射到 `postData` 条目；并做去重、缺失容错。
   - 约束：selector 必须 SSR 安全（不访问 `window/localStorage`）。

5. **新增首页模块组件（复用现有样式原子）**
   - 新建组件（建议路径）：`.vitepress/theme/components/Home/HomeHighlights.vue`。
   - 结构建议：两列网格（desktop 2col / mobile 1col），每列为一个 `s-card`，内部为简洁列表（可复用 `PostList` 的 `simple` 模式，或自建 `MiniPostList` 以精确控制密度）。
   - 行为建议：
     - 每个板块标题栏包含“查看全部”链接（Recent → `/pages/archives`；Popular → 可跳转到某个 tag/category/自定义页）。
     - 列表项点击 `router.go(item.regularPath)`。

6. **将模块接入首页（仅对首页首屏生效）**
   - 在 `.vitepress/theme/views/Home.vue` 内插入 `<HomeHighlights />`，位置建议在 `TypeBar` 之前。
   - 增加 `shouldShowHighlights` 计算条件：`!showCategories && !showTags && getCurrentPage() === 0`。

7. **样式实现策略（保持与现有系统一致）**
   - 继续使用组件内 `scoped lang="scss"`；布局使用 `display: grid` + `gap: 1rem`。
   - 颜色/边框/阴影全部使用现有 CSS 变量与 `.s-card`（避免引入新的设计系统）。
   - 断点对齐现有规则：`1200px`（Aside 隐藏阈值）与 `768px`（移动端）。

8. **验证与回归（确保不破坏分类/分页/暗色模式）**
   - 本地验证路径：`/`、`/page/2`、`/pages/categories/<name>`、`/pages/tags/<name>`。
   - 检查暗色模式（`html.dark`）下对比度与 hover 态。
   - 构建验证：`npm run build`（重点关注 SSR 报错与 selector 的纯函数约束）。

⚠️ 风险与注意事项
- **“Most Popular”缺少真实数据源**：当前仓库内 `postData` 不包含 PV/热度字段；文章页热度展示来自运行时（Artalk/Twikoo）注入，无法直接用于构建期排序。
- **外部统计依赖的构建不稳定**：若构建期拉取 PV 排行，需缓存与超时降级策略，否则 CI/本地 build 易失败。
- **重复内容与信息密度**：Recent 与 Popular 可能重复；需要去重与展示条数控制。
- **SSR 安全**：首页模块与 selector 不能使用 `window`（分类分页逻辑里已有 `window` 分支，新增模块要避免踩坑）。

📎 参考
- `index.md:1`（首页入口，layout=home）
- `.vitepress/theme/views/Home.vue:1`（首页布局与插入点）
- `.vitepress/theme/utils/getPostData.mjs:33`（文章排序规则：top + date）
- `.vitepress/config.mjs:17`（构建期生成并注入 themeConfig.postData）
- `.vitepress/theme/style/main.scss:200`（全局卡片样式 `.s-card` 与变量体系）
- `.vitepress/theme/views/Post.vue:41`（文章页“热度/PV”展示为运行时注入）
