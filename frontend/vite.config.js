import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// During `npm run dev` we proxy /api/* to the FastAPI backend so the
// client can use a single relative '/api' baseURL in both dev and prod.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
