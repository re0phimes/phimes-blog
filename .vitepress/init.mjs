import { themeConfig } from "./theme/assets/themeConfig.mjs";
import { existsSync } from "fs";
import path from "path";
import { fileURLToPath } from "url";

// 注意：这是一个 ESM 文件，没有 __dirname。
// 之前这里用的是 __dirname，在纯 Node 环境下是 undefined，
// path.resolve(undefined, ...) 会抛异常 → 被 catch 吃掉 →
// 静默回退到默认配置（只留一行 console.warn）。
// VitePress 打包时会 shim __dirname，所以在站点里看不出问题；
// 但 page/[num].paths.mjs 这类 data loader 一旦拿不到用户配置，
// 分页数就会按默认 postSize 算，错得很隐蔽。
const __dirname = path.dirname(fileURLToPath(import.meta.url));

/**
 * 获取并合并配置文件
 */
export const getThemeConfig = async () => {
  try {
    // 配置文件绝对路径
    const configPath = path.resolve(__dirname, "../themeConfig.mjs");
    if (existsSync(configPath)) {
      // 文件存在时进行动态导入
      const userConfig = await import("../themeConfig.mjs");
      return Object.assign(themeConfig, userConfig?.themeConfig || {});
    }
    // 文件不存在时返回默认配置
    console.warn("User configuration file not found, using default themeConfig.");
    return themeConfig;
  } catch (error) {
    // 这里是真正的异常（比如用户配置语法错误），不应该只留一行日志
    console.error("An error occurred while loading the configuration:", error);
    return themeConfig;
  }
};
