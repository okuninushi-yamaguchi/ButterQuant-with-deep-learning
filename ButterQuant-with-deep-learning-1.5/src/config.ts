export const config = {
  // 使用空字符串让请求走 Vite 代理 (转发到 5001)
  API_URL: import.meta.env.VITE_API_URL || '',
};
