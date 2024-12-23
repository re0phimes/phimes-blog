import { mainStore } from './store'

export default {
  enhanceApp({ app }) {
    // 初始化主题
    const store = mainStore()
    store.initTheme()
  }
} 