import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    // 本機 `npm run dev` 時把 /api 代理到本機後端,避免 CORS
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
