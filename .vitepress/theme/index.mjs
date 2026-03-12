import { h } from "vue";
import { createPinia } from "pinia";
import { routeChange } from "@/utils/initTools.mjs";
import { enhanceAppWithTabs } from "vitepress-plugin-tabs/client";
import LazyLoader from "@/components/LazyLoader.vue";
import piniaPluginPersistedstate from "pinia-plugin-persistedstate";
import { themeConfig } from "../../themeConfig.mjs";

// 根组件
import App from "@/App.vue";
// 全局样式
import "@/style/main.scss";
import "katex/dist/katex.min.css";

// pinia
const pinia = createPinia();
pinia.use(piniaPluginPersistedstate);

let InstantSearch = null;
if (themeConfig?.search?.enable) {
  ({ default: InstantSearch } = await import("vue-instantsearch/vue3/es"));
}

// Theme
const Theme = {
  // extends: Theme,
  Layout: () => {
    return h(App);
  },
  enhanceApp({ app, router, siteData }) {
    // 挂载
    app.use(pinia);
    if (InstantSearch && siteData?.value?.themeConfig?.search?.enable) {
      app.use(InstantSearch);
    }
    app.component("LazyLoader", LazyLoader);
    // 插件
    enhanceAppWithTabs(app);
    // 路由守卫
    router.onBeforeRouteChange = (to) => {
      routeChange("before", to);
    };
    router.onAfterRouteChanged = (to) => {
      routeChange("after", to);
    };
  },
};

export default Theme;
