---
mode: plan
cwd: /home/modelenv/chentianxuan/s_projects/phimes-blog
task: 修复首页热门标签遮挡 Highlights，并扩展页面宽度（减少左右空白）
complexity: medium
planning_method: builtin
created_at: 2025-12-31T11:46:47+08:00
---

# Plan: 扩展布局宽度 + 解决首页遮挡

🎯 任务概述
目前首页新增的 `Most Popular / Recent Posts` 模块在某些屏幕宽度/内容组合下，会被右侧栏的「热门标签」区域视觉遮挡（或发生横向溢出导致覆盖）。同时当前全站主内容区采用 `max-width` 居中策略，用户期望在大屏下更“铺开”，减少左右留白，提高左右占用。

本计划目标：在不引入新依赖、不破坏现有首页/分类/标签/文章页结构的前提下，定位遮挡根因并通过“布局宽度策略 + 侧边栏标签溢出治理”两条线闭环解决。

📋 执行计划
1. **复现与定位遮挡根因（先证据后改动）**
   - 用浏览器在首页 `/` 复现：记录出现遮挡时的 viewport 宽度区间、是否触发滚动/粘性（sticky）状态、具体是哪一块覆盖（`aside .tags-cloud` vs `HomeHighlights`）。
   - DevTools 检查：确认是否为 flex 宽度计算问题（布局真的重叠）还是标签项过长导致横向溢出（单个 `.tags` 超出 aside 宽度并覆盖左侧）。
   - 输出一条“根因结论 + 复现口径”（写入 issue/notes 或本 plan 的修订记录），确保后续改动可验收。

2. **确定“页面更宽”的目标策略（选择一条可回滚路线）**
   - 方案 A（推荐，低风险）：把全站主容器 `max-width` 从 1400 提升到更大（例如 1600/1800），仍然居中；兼顾阅读性与“更铺开”诉求。
   - 方案 B（更激进）：取消/显著放宽 `max-width`，改为 `width: 100%` + 更合理的左右 padding（例如 3–6rem），在超大屏下更接近“满屏杂志”。
   - 确定是否需要同步调整 Nav/Footer 的 `max-width` 以保持对齐一致（避免“导航很窄/正文很宽”的割裂）。

3. **扩展全站内容区宽度（保证组件间不再挤压）**
   - 修改 `.mian-layout` 的布局宽度策略（`max-width` / `padding`），并验证首页、文章页、普通页面的左右留白符合预期。
   - 同步调整 `Nav` 的容器宽度上限与左右 padding，让头部与正文对齐。
   - 评估是否需要同步调整 Footer / FooterLink 的宽度上限，保持整体一致性（若不调整需确认视觉上可接受）。

4. **治理「热门标签」的横向溢出/遮挡（确保永不覆盖左侧内容）**
   - 给 `.tags-cloud .all-tags .tags` 增加可收缩约束（`max-width: 100%`、`min-width: 0`、必要时改为 `inline-flex`），避免长 tag 名称把 flex item 撑破。
   - 给 `.name` 增加截断或断行策略（二选一）：
     - 截断：`overflow: hidden; text-overflow: ellipsis; white-space: nowrap;`（更紧凑）。
     - 断行：`overflow-wrap: anywhere;`（信息更完整但更高）。
   - 验证包含长 tag（如带逗号/空格）时不会产生横向滚动条、更不会覆盖 `posts-content`。

5. **（如仍存在）优化首页左右布局计算与间距（减少“视觉挤压”）**
   - 把 Home 页的 `.home-content` 布局从“写死宽度 calc”改为更稳健的 flex（例如 `posts-content: flex: 1; min-width: 0;` + `aside: flex: 0 0 300px;` + `gap`），避免未来组件变更导致重叠。
   - 视情况把 `HomeHighlights` 的两列断点提前（例如在更窄的 viewport 下直接 1 列），防止在“内容区仍偏窄”的场景下卡片过挤。

6. **回归验证与验收记录（可复现、可对比）**
   - 手工回归路由：`/`、`/page/2`、`/pages/tags/*`、`/pages/categories/*`、文章页任意 3 篇；在 768/1024/1200/1400/1600+ 等宽度检查布局。
   - 检查点：首页 Highlights 不被遮挡；Aside 标签不横向溢出；无横向滚动条；暗色模式对比度与 hover 正常。
   - 构建验证：`npm run build` 通过（SSR 无报错）；如项目 lint 可用则补跑，否则记录受限原因与后续修复建议。

⚠️ 风险与注意事项
- **全局宽度变更的连锁影响**：修改 `.mian-layout`/Nav/Footer 的宽度可能影响所有页面的视觉密度；需要明确“只放宽多少”的可回滚方案。
- **标签溢出策略的取舍**：截断会隐藏部分 tag 名称；断行会增高 aside，高度变化可能影响 sticky 的可用性，需要在信息完整性与视觉稳定性之间取舍。
- **断点不一致**：Home、Post、Page 都有 1200/768 的断点逻辑，调整需避免出现某些页面“该隐藏却没隐藏/该一列却两列”的割裂体验。

📎 参考
- `.vitepress/theme/App.vue:157`（主内容区 `.mian-layout` 的 max-width/padding）
- `.vitepress/theme/components/Nav.vue:212`（头部容器宽度上限与 padding）
- `.vitepress/theme/components/Footer.vue:84`（Footer 内容区 max-width）
- `.vitepress/theme/components/FooterLink.vue:59`（FooterLink max-width=1200 的潜在不一致）
- `.vitepress/theme/views/Home.vue:172`（首页左右布局：posts-content 与 main-aside）
- `.vitepress/theme/components/Aside/Widgets/Tags.vue:28`（热门标签样式：可能的横向溢出根因）

