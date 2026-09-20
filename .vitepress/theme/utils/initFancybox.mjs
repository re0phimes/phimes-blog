/**
 * 图片灯箱初始化
 *
 * 之前是从第三方 CDN（mirrors.sustech.edu.cn）动态拉 fancybox 的 js/css。
 * 实测线上那个地址在浏览器里取不到（ERR_CONNECTION_CLOSED / ERR_BLOCKED_BY_ORB），
 * window.Fancybox 一直是 undefined，所以「点击图片放大」全站失效。
 *
 * 现在改用 npm 依赖 @fancyapps/ui，由 Vite 打进产物，不再依赖外部 CDN。
 */

const FANCYBOX_OPTIONS = {
  hideScrollbar: true,
  Carousel: {
    transition: "slide",
  },
  Hash: false,
  Toolbar: {
    display: {
      left: ["infobar"],
      middle: ["zoomIn", "zoomOut", "toggle1to1", "rotateCCW", "rotateCW", "flipX", "flipY"],
      right: ["slideshow", "thumbs", "close"],
    },
  },
  l10n: {
    PANUP: "上移",
    PANDOWN: "下移",
    PANLEFT: "左移",
    PANRIGHT: "右移",
    ZOOMIN: "放大",
    ZOOMOUT: "缩小",
    TOGGLEZOOM: "切换缩放级别",
    TOGGLE1TO1: "切换缩放级别",
    ITERATEZOOM: "切换缩放级别",
    ROTATECCW: "逆时针旋转",
    ROTATECW: "顺时针旋转",
    FLIPX: "水平翻转",
    FLIPY: "垂直翻转",
    FITX: "水平适应",
    FITY: "垂直适应",
    RESET: "重置",
    TOGGLEFS: "切换全屏",
    CLOSE: "关闭",
    NEXT: "下一个",
    PREV: "上一个",
    MODAL: "使用 ESC 键关闭",
    ERROR: "发生了错误，请稍后再试",
    IMAGE_ERROR: "找不到图像",
    ELEMENT_NOT_FOUND: "找不到 HTML 元素",
    AJAX_NOT_FOUND: "载入 AJAX 时出错: 未找到",
    AJAX_FORBIDDEN: "载入 AJAX 时出错: 被阻止",
    IFRAME_ERROR: "加载页面出错",
    TOGGLE_ZOOM: "切换缩放级别",
    TOGGLE_THUMBS: "切换缩略图",
    TOGGLE_SLIDESHOW: "切换幻灯片",
    TOGGLE_FULLSCREEN: "切换全屏",
    DOWNLOAD: "下载",
  },
};

const initFancybox = async (themeConfig) => {
  try {
    if (!themeConfig?.fancybox?.enable) return false;
    if (typeof document === "undefined") return false;
    if (!document.querySelector("[data-fancybox]")) return false;

    // 动态引入，避免 SSR 阶段碰到 DOM
    await import("@fancyapps/ui/dist/fancybox/fancybox.css");
    const { Fancybox } = await import("@fancyapps/ui");

    Fancybox.bind("[data-fancybox]", FANCYBOX_OPTIONS);
    return true;
  } catch (error) {
    console.error("图片灯箱初始化失败", error);
    return false;
  }
};

export default initFancybox;
