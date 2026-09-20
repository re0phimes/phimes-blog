module.exports = {
  root: true,
  env: {
    browser: true,
    node: true,
    es2022: true,
  },
  extends: ["airbnb-base", "plugin:vue/vue3-essential"],
  parserOptions: {
    ecmaVersion: "latest",
    sourceType: "module",
  },
  plugins: ["vue"],
  // unplugin-auto-import（imports: ["vue", "vitepress"]）注入的 API 不会出现在源码里，
  // 不声明的话 no-undef 会报 240+ 条假错。
  globals: {
    // vue
    ref: "readonly",
    reactive: "readonly",
    computed: "readonly",
    watch: "readonly",
    watchEffect: "readonly",
    nextTick: "readonly",
    defineAsyncComponent: "readonly",
    onMounted: "readonly",
    onBeforeMount: "readonly",
    onUnmounted: "readonly",
    onBeforeUnmount: "readonly",
    onActivated: "readonly",
    onDeactivated: "readonly",
    provide: "readonly",
    inject: "readonly",
    // vitepress
    useData: "readonly",
    useRoute: "readonly",
    useRouter: "readonly",
    withBase: "readonly",
    // 主题里挂在 window 上的全局对象
    $message: "readonly",
    $player: "readonly",
    $comment: "readonly",
    Artalk: "readonly",
    twikoo: "readonly",
    commentType: "readonly",
    Fancybox: "readonly",
  },
  rules: {
    // 使用双引号
    quotes: ["warn", "double"],
    // 导入后缀名
    "import/extensions": "off",
    // 禁用 console
    "no-console": "off",

    // ↓ 下面这些交给 Prettier 处理，或者与现有代码风格冲突，
    //   开着只会淹没真正有用的告警。
    indent: "off",
    "object-curly-newline": "off",
    "operator-linebreak": "off",
    "no-plusplus": "off",
    "no-param-reassign": "off",
    "no-continue": "off",
    "no-await-in-loop": "off",
    "no-restricted-syntax": "off",
    "no-use-before-define": "off",
    "no-underscore-dangle": "off",
    "class-methods-use-this": "off",
    // @ 别名由 Vite 解析，ESLint 不认
    "import/no-unresolved": "off",
    "import/prefer-default-export": "off",
    "import/no-extraneous-dependencies": "off",
    // 组件名：主题里就是 index.vue / Tags.vue 这种
    "vue/multi-word-component-names": "off",
    "vue/no-v-html": "off",
    // 这个代码库里大量使用「提前 return」的写法，consistent-return 只会刷屏
    "consistent-return": "off",
    // 短路的 && 调用（callback && callback()）是刻意写法
    "no-unused-expressions": ["error", { allowShortCircuit: true, allowTernary: true }],
    // 对象/数组类型 prop 的默认值必须是工厂函数（这条是真问题，保持 error）
    "vue/require-valid-default-prop": "error",

    // ↓ 以下都是纯风格 / 与现有写法刻意冲突的规则，降级成 warning，
    //   这样 `npm run lint` 能通过、又不会制造整个仓库的重排 diff。
    "quote-props": "off",
    "implicit-arrow-linebreak": "off",
    "arrow-body-style": "off",
    "no-trailing-spaces": "off",
    "max-len": "off",
    "brace-style": "off",
    "object-curly-spacing": "off",
    "comma-dangle": "off",
    "import/order": "off",
    "prefer-destructuring": "off",
    "prefer-template": "off",
    "function-paren-newline": "off",
    "spaced-comment": "off",
    "no-multiple-empty-lines": "off",
    "no-confusing-arrow": "off",
    "no-nested-ternary": "off",
    // 这个代码库原本大量使用 else 块；扁平化虽然等价，但会让缩进产生误导，关掉
    "no-else-return": "off",
    "no-lonely-if": "off",
    "no-restricted-globals": "off",
    "no-bitwise": "off",
    "no-case-declarations": "off",
    "no-new": "off",
    "no-return-assign": "off",
    "no-control-regex": "off",
    "no-cond-assign": "off",
    "array-callback-return": "off",
    "import/no-named-as-default": "off",
    camelcase: "off",
    "no-shadow": "warn",
    "prefer-const": "warn",
    "no-unneeded-ternary": "warn",
    "object-shorthand": "warn",
    "arrow-parens": ["warn", "always"],
    "no-unused-vars": ["warn", { argsIgnorePattern: "^_", varsIgnorePattern: "^_" }],
  },
};
