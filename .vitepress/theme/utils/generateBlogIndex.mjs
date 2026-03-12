import { writeFileSync } from "fs";
import path from "path";
import { getAllPosts } from "./getPostData.mjs";
import { getPostPublicPath } from "./postUrl.mjs";

/**
 * 生成博客文章索引 JSON
 * @param {*} config VitePress buildEnd config
 * @param {*} themeConfig 主题配置
 */
export const createBlogIndex = async (config, themeConfig) => {
  const hostLink = themeConfig.siteMeta.site;
  let posts = await getAllPosts();

  // 日期降序排序
  posts = posts.sort((a, b) => {
    const dateA = new Date(a.date);
    const dateB = new Date(b.date);
    return dateB - dateA;
  });

  const index = posts.map((post) => ({
    title: post.title,
    url: `${hostLink}${getPostPublicPath(post)}`,
  }));

  writeFileSync(
    path.join(config.outDir, "index.json"),
    JSON.stringify(index, null, 2),
    "utf-8",
  );
};
