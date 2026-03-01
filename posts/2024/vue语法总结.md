---
title: vue语法总结 2024-12-23
tags: [vue, 前端]
topic: [vue, syntax, frontend]
categories: [技术总结]
date: 2024-10-10
# description: 欢迎使用 Curve 主题，这是你的第一篇文章
articleGPT: 这是一篇vue语法总结，记录了vue的一些关键语法和使用方法
# references:
#   - icon: <i class="fa-brands fa-github" />
#     name: Github
#   - title: vitepress-theme-curve
#     url: https://github.com/phi/phimes-blog
# friends:
#   - Baka-Monogatari
#   - imsyy
---

# 特殊语法
## 主要的v-语法

vue实现了一系列v开头指令，这些指令

```
v-if="条件"        // 条件渲染
v-else-if="条件"   // 条件渲染
v-else            // 条件渲染
v-show="条件"      // 显示/隐藏 (通过CSS)
v-for="item in items"  // 列表渲染
v-on:click="方法"    // 事件绑定，简写为 @click
v-bind:prop="值"    // 属性绑定，简写为 :prop
v-model="变量"      // 双向绑定
v-html="html内容"   // 渲染HTML
v-text="文本"      // 渲染文本
```


## 属性绑定 v-bind

`:(v-bind)` 是属性绑定，单向的数据流绑定，属性绑定可以分为静态绑定和动态（属性名）绑定两种。

### 静态绑定
v-bind可以绑定HTML的标准属性也可以自定义属性的绑定，我们可以用冒号`:`进行缩写。
**HTML标准属性**`
```vue 
<!-- HTML标准属性 -->
<div :id="myId">              // DOM id属性
<div :class="myClass">        // class属性
<div :style="myStyle">        // style属性
<img :src="imgUrl">           // src属性
<input :type="inputType">     // type属性
<a :href="link">             // href属性
```
在HTML中标准属性中我们会出现直接使用，也会出现绑定变量属性的情况
**HTML标准绑定对比**
```vue file="绑定属性例子.vue"
<template>
  <div>
    <!-- 1. 图片链接 -->
    <img src="/static/logo.png"/>      <!-- 静态图片 -->
    <img :src="userAvatar"/>           <!-- 动态图片 -->

    <!-- 2. class绑定 -->
    <div class="btn">                  <!-- 静态class -->
    <div :class="{ 
      'active': isActive,              <!-- 动态class -->
      'disabled': isDisabled 
    }">

    <!-- 3. style绑定 -->
    <div style="color: red">           <!-- 静态样式 -->
    <div :style="{                     <!-- 动态样式 -->
      color: textColor,
      fontSize: fontSize + 'px'
    }">

    <!-- 4. 表单元素 -->
    <input type="text" value="静态值">  <!-- 静态值 -->
    <input :value="dynamicValue">      <!-- 动态值 -->

    <!-- 5. 禁用状态 -->
    <button disabled>静态禁用</button>   <!-- 静态禁用 -->
    <button :disabled="isDisabled">    <!-- 动态禁用 -->
      提交
    </button>

    <!-- 6. 自定义属性 -->
    <div data-id="1">                  <!-- 静态数据属性 -->
    <div :data-id="userId">           <!-- 动态数据属性 -->

    <!-- 7. href链接 -->
    <a href="/about">关于</a>          <!-- 静态链接 -->
    <a :href="dynamicLink">详情</a>     <!-- 动态链接 -->
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

// 响应式数据
const userAvatar = ref('/images/avatar.jpg')
const isActive = ref(true)
const isDisabled = ref(false)
const textColor = ref('blue')
const fontSize = ref(16)
const dynamicValue = ref('Hello Vue')
const userId = ref(123)
const dynamicLink = ref('/user/profile')

// 计算属性示例
const buttonClass = computed(() => ({
  'btn': true,
  'btn-active': isActive.value,
  'btn-disabled': isDisabled.value
}))
</script>
```

**自定义属性**
```
<!-- 完全自定义的属性名 -->
<div :my-custom="value">      // 自定义属性名
<div :data-test="value">      // 自定义data属性
<div :whatever="value">       // 任意命名的属性

<!-- 在组件上使用 -->
<MyComponent :custom-prop="value">  // 自定义props
<MyComponent :any-name="value">     // 任意命名的props
```

### 动态（属性名）绑定

```vue file="动态属性exmaple.vue"
<template>
  <!-- 属性名本身也可以是动态的 -->
  <div :[dynamicAttrName]="value">
    动态属性名
  </div>
</template>

<script>
export default {
  data() {
    return {
      dynamicAttrName: 'custom-attr', // 可以动态改变属性名
      value: 'some value'
    }
  }
}
</script>
```

**动态属性名例子**

```vue file="动态属性.vue"
<template>
  <div>
    <!-- 动态数据属性 -->
    <div :[dataAttribute]="dataValue">
      数据属性展示
    </div>

    <!-- 动态aria属性 -->
    <button :[ariaAttribute]="ariaValue">
      无障碍按钮
    </button>

    <!-- 动态样式属性 -->
    <div :[styleAttribute]="styleValue">
      样式测试
    </div>

    <!-- 控制面板 -->
    <div class="controls">
      <button @click="changeDataAttr">切换数据属性</button>
      <button @click="changeAriaAttr">切换无障碍属性</button>
      <button @click="changeStyleAttr">切换样式属性</button>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

// 数据属性相关
const dataAttribute = ref('data-test')
const dataValue = ref('test-value')

// 无障碍属性相关
const ariaAttribute = ref('aria-label')
const ariaValue = ref('可访问按钮')

// 样式属性相关
const styleAttribute = ref('style')
const styleValue = ref('color: red')

// 切换方法
const changeDataAttr = () => {
  if (dataAttribute.value === 'data-test') {
    dataAttribute.value = 'data-id'
    dataValue.value = 'id-123'
  } else {
    dataAttribute.value = 'data-test'
    dataValue.value = 'test-value'
  }
}

const changeAriaAttr = () => {
  if (ariaAttribute.value === 'aria-label') {
    ariaAttribute.value = 'aria-description'
    ariaValue.value = '这是一个按钮描述'
  } else {
    ariaAttribute.value = 'aria-label'
    ariaValue.value = '可访问按钮'
  }
}

const changeStyleAttr = () => {
  if (styleValue.value.includes('red')) {
    styleValue.value = 'color: blue; font-size: 18px'
  } else {
    styleValue.value = 'color: red; font-size: 16px'
  }
}
</script>
```

### 父组件给子组件传递绑定属性
属性绑定也通常用于父子组件的通信传递[[vue总结#组件间通信]]，可以在组件通信章节查看具体方式


## 事件绑定


# 组件间通信

### 父子组件通信

通常的父子组件通信可以使用`DefineProps`和`DefineEmit`，一个用于父组件向字组件传递，一个用于子组件向父组件传递。这不仅可以传递的属性，也可以传递方法的结果。如果更为复杂的场景可以使用`pinia`[[#使用Pinia]]。


> 父子组件通信可以使用DefineProps来获取的props对象，从而使用父组件传递的变量

**父组件**

```vue file="父子组件通信-通过DefineProps-父组件.vue"
<!-- 父组件 -->
<template>
  <ChildComponent
    :user-name="name"
    :age="age"
    :is-admin="isAdmin"
    :config="config"
  />
</template>

<script setup>
import { ref, reactive } from 'vue'

const name = ref('Tom')
const age = ref(25)
const isAdmin = ref(true)
const config = reactive({
  theme: 'dark',
  showHeader: true
})
</script>
```

**子组件**

```vue file="子组件.vue"
<!-- 子组件 -->
<template>
  <div>
    <h2>用户信息</h2>
    <p>姓名: {{ userName }}</p>
    <p>年龄: {{ age }}</p>
    <p>管理员: {{ isAdmin ? '是' : '否' }}</p>
    <p>主题: {{ config.theme }}</p>
    
    <!-- 使用计算属性 -->
    <p>状态: {{ userStatus }}</p>
  </div>
</template>

<script setup>
import { computed } from 'vue'

// 声明所需的全部props
const props = defineProps({
  userName: {
    type: String,
    required: true
  },
  age: {
    type: Number,
    default: 0,
    validator: (value) => value >= 0
  },
  isAdmin: {
    type: Boolean,
    default: false
  },
  config: {
    type: Object,
    default: () => ({
      theme: 'light',
      showHeader: false
    })
  }
})

// 基于props的计算属性
const userStatus = computed(() => {
  if (props.isAdmin) return '管理员'
  return props.age > 18 ? '成年用户' : '未成年用户'
})

// 使用解构来简化访问
// 注意：解构会失去响应性
const { userName, age } = props

// 如果需要保持响应性，可以使用toRefs
import { toRefs } from 'vue'
const { userName, age } = toRefs(props)
</script>
```

### 子组件中三中获取属性的方式

在子组件中，其实有三种方式去使用父组件传递的变量。
- DefineProps后用`props.xxx`来获取
- 直接解构props，获取所有属性，失去响应性。
- 通过`toRef`解构，保持变量响应性

```vue file="传递变量的方式.vue"
<!-- 父组件 -->
<template>
  <div>
    <child-component :count="count" />
    <button @click="increment">增加</button>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const count = ref(0)
const increment = () => {
  count.value++
}
</script>

<!-- 子组件 -->
<script setup>
import { toRefs } from 'vue'

const props = defineProps({
  count: Number
})

// 1. 直接使用props (✅ 有响应性)
console.log(props.count)  // 会随父组件变化而变化

// 2. 直接解构 (❌ 失去响应性)
const { count: countNormal } = props
console.log(countNormal)  // 只是一个普通数值,不会随父组件变化

// 3. 使用toRefs解构 (✅ 保持响应性)
const { count: countRef } = toRefs(props)
console.log(countRef.value)  // 会随父组件变化而变化
</script>

<template>
  <div>
    <!-- 这三种方式的显示效果: -->
    <p>直接使用props: {{ props.count }}</p>      <!-- 会更新 -->
    <p>直接解构: {{ countNormal }}</p>           <!-- 不会更新,保持初始值 -->
    <p>toRefs解构: {{ countRef }}</p>            <!-- 会更新 -->
  </div>
</template>
```

> 既然方法一通过props的属性的方式获取已经足够了为什么还要解构?

主要是因为：
- 可以简化代码，不用次次都写props
- 可以配合ts获取更好的推导
- 可以更好的和其他API配合使用
- 业务上或许只需要一次属性


#### 简化逻辑

```
<script setup>
const props = defineProps({
  firstName: String,
  lastName: String,
  age: Number,
  address: String
})

// 方式1：直接使用props
const fullName = computed(() => {
  return `${props.firstName} ${props.lastName}, ${props.age}岁, 住在${props.address}`
})

// 方式2：使用toRefs解构
const { firstName, lastName, age, address } = toRefs(props)
const fullName = computed(() => {
  return `${firstName.value} ${lastName.value}, ${age.value}岁, 住在${address.value}`
})
</script>
```

#### 重复使用

```
<script setup>
const props = defineProps({
  userConfig: Object
})

// 方式1：每次都要写props
const handleClick = () => {
  if (props.userConfig.isAdmin) {
    doSomething(props.userConfig.permissions)
    checkAccess(props.userConfig.level)
    updateUI(props.userConfig.theme)
  }
}

// 方式2：解构后更简洁
const { userConfig } = toRefs(props)
const handleClick = () => {
  if (userConfig.value.isAdmin) {
    doSomething(userConfig.value.permissions)
    checkAccess(userConfig.value.level)
    updateUI(userConfig.value.theme)
  }
}
```

#### 类型推导

```
<script setup lang="ts">
interface UserProps {
  name: string
  age: number
  settings: {
    theme: string
    notifications: boolean
  }
}

const props = defineProps<UserProps>()

// 使用解构，可以直接获得类型提示
const { settings } = toRefs(props)
// settings.value 会有完整的类型提示
```

#### 其他API组合使用

```
<script setup>
import { watch } from 'vue'

const props = defineProps(['count'])

// 方式1：直接使用props
watch(
  () => props.count,
  (newVal) => {
    console.log(newVal)
  }
)

// 方式2：使用toRefs解构
const { count } = toRefs(props)
// 可以直接监听解构出来的ref
watch(count, (newVal) => {
  console.log(newVal)
})
```

### 使用Pinia
