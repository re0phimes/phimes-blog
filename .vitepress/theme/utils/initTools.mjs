import { mainStore } from "@/store";

/**
 * 判断是否即将导航到的地址和当前地址是相同页面
 * @return {boolean} 为 true 时表示是相同页面
 */
export const isSamePage = (to) => {
  if (typeof window === "undefined") return false;
  // 获取跳转到的页面路径
  const toURL = new URL(to, window.location.origin);
  const targetPathWithoutHash = toURL.pathname;
  // 获取当前页面的路径
  const currentURL = new URL(window.location.href);
  const currentPathWithoutHash = currentURL.pathname;
  return targetPathWithoutHash === currentPathWithoutHash;
};

// 路由跳转前
export const routeChange = (type, to) => {
  if (typeof window === "undefined") return false;
  if (type === "before") {
    if (!isSamePage(to)) {
      changeLoading({ status: true });
    }
    return false;
  }

  if (type === "after") {
    changeLoading({ status: false });
  }
};

// 切换加载状态
const changeLoading = (option = {}) => {
  const store = mainStore();
  const { status = true } = option;
  store.loadingStatus = status;
};
