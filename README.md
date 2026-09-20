<h1 align="center"> phimes-blog </h1>

<p align="center">在 AI 时代思考与分享 · 基于 VitePress 的个人博客</p>

<p align="center">
  <a href="https://blog.phimes.top/">线上站点</a> ·
  <a href="https://www.phimes.top/">主站</a> ·
  <a href="https://aifaq.phimes.top/">AI FAQ</a> ·
  <a href="https://demo.phimes.top/">Demo</a>
</p>

---

> [!NOTE]
> 本站主题最初 fork 自 [vitepress-theme-blog](https://github.com/imsyy/vitepress-theme-blog)，
> 之后做了较大改造：亮色主题换成 Anthropic 配色、首页改成「头条 + 历史文章（可排序/筛选）」、
> 数学公式改用 katex 0.16 并修掉了渲染问题、支持 Obsidian 双链语法等。

## 本地开发

```bash
npm install
npm run dev        # 开发服务器（默认 http://localhost:9877）
npm run build      # 生产构建 → .vitepress/dist
npm run preview    # 预览构建产物
npm run lint       # ESLint
npm run format     # Prettier
```

构建前会先跑 `npm run fetch-popular`，把文章浏览量抓到 `data/popular.json`，
首页「按热度排序」依赖这份数据。**该文件是构建产物，已在 .gitignore 中。**

## 目录结构

```
.vitepress/
  config.mjs                 # VitePress 站点配置
  init.mjs                   # 合并 themeConfig（用户配置覆盖默认配置）
  theme/
    index.mjs                # 主题入口
    assets/themeConfig.mjs   # 主题默认配置（不要直接改这个）
    themeConfig.mjs -> 根目录 # 站点自己的配置
    components/              # 组件（由 unplugin-vue-components 自动按目录导入）
    views/                   # 页面级组件（首页/文章/归档…）
    utils/                   # markdown 插件、构建期工具
    style/                   # 全局样式（main.scss / post.scss）
    composables/             # 组合式函数
    store/                   # Pinia store
posts/                       # 文章（markdown + frontmatter）
pages/                       # 独立页面
page/[num].md                # 分页跳板（→ /?p=N）
public/                      # 静态资源（covers 为构建期下载的封面）
scripts/                     # 构建脚本
tests/                       # 单元测试（node --test）
```

## 写文章

在 `posts/` 下新建 markdown 文件，frontmatter 示例：

```yaml
---
title: 文章标题
tags: [llm, transformer]
categories: [llm-principles]
date: 2026-01-16
cover: https://image.phimes.top/img/xxx.png
---
```

- 图片统一放 `image.phimes.top`（阿里云 OSS 的 CNAME）。该 bucket 配了**防盗链白名单**，
  白名单里包含 `http://127.0.0.1:*` 和 `http://localhost:*`，所以本地开发也能正常显示。
- 支持 `[[笔记名]]` 这种 Obsidian 双链语法，会解析成站内链接（见 `utils/markdownWikilink.mjs`）。
- 数学公式用 `$...$` / `$$...$$`，由 `utils/markdownKatex.mjs` 渲染（katex 0.16）。

## 配置

站点配置在仓库根目录的 `themeConfig.mjs`，它通过 `Object.assign` **浅合并**覆盖
`.vitepress/theme/assets/themeConfig.mjs` 里的默认值。

> ⚠️ 因为是浅合并，**任何嵌套对象只要在用户配置里写了一半，就会整个替换掉默认值**。
> 例如你写 `aside: { tags: {...} }`，那么 `aside.toc`、`aside.siteData` 就都没了。
> 改动 `aside` / `home` 这类嵌套结构时记得把完整的键写全。

## 部署

推送到 `master` 后由 Vercel 自动构建部署。另外有一个 GitHub Actions
（`.github/workflows/trigger-faq-rebuild.yml`）会在推送后触发 AI FAQ 站点的重建。

## 测试

```bash
node --test tests/
```

## License

MIT
