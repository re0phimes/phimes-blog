import { createContentLoader } from "vitepress";
import { writeFileSync } from "fs";
import path from "path";

/**
 * 生成博客文章索引 JSON
 * @param {*} config VitePress buildEnd config
 * @param {*} themeConfig 主题配置
 */
export const createBlogIndex = async (config, themeConfig) => {
  const hostLink = themeConfig.siteMeta.site;
  let posts = await createContentLoader("posts/**/*.md").load();

  // 日期降序排序
  posts = posts.sort((a, b) => {
    const dateA = new Date(a.frontmatter.date);
    const dateB = new Date(b.frontmatter.date);
    return dateB - dateA;
  });

  const index = posts.map(({ url, frontmatter }) => ({
    title: frontmatter.title,
    url: `${hostLink}${url}`,
  }));

  writeFileSync(
    path.join(config.outDir, "index.json"),
    JSON.stringify(index, null, 2),
    "utf-8",
  );
};
