/**
 * 友情链接数据
 *
 * 原来这里是上游主题作者（無名小栈）的友链列表，头像全部指向他已经停用的
 * 图床 pic.efefee.cn（域名已无法解析），所以整页头像都是碎的。
 *
 * 现在改成空列表，请按下面的结构填自己的友链。
 * 头像建议放自己的图床（image.phimes.top）或本站 public/ 下，避免再依赖第三方。
 *
 * 分组结构：
 *   {
 *     type: "friends",           // 分组标识（"loss" 表示"失联"分组，卡片不可点）
 *     typeName: "小伙伴们",       // 分组标题
 *     typeDesc: "我们在一起，共同进步",  // 分组副标题
 *     typeList: [
 *       {
 *         name: "某某的博客",
 *         avatar: "https://image.phimes.top/img/xxx.png",
 *         desc: "一句话介绍",
 *         url: "https://example.com/",
 *       },
 *     ],
 *   }
 */
const linkData = [
  // 示例（把注释去掉并改成自己的友链即可）：
  // {
  //   type: "friends",
  //   typeName: "小伙伴们",
  //   typeDesc: "我们在一起，共同进步",
  //   typeList: [
  //     {
  //       name: "某某的博客",
  //       avatar: "https://image.phimes.top/img/avatar.png",
  //       desc: "一句话介绍",
  //       url: "https://example.com/",
  //     },
  //   ],
  // },
];

export default linkData;
