import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/simulate':     'http://localhost:8000',
      '/classify':     'http://localhost:8000',
      '/engine':       'http://localhost:8000',
      '/fault':        'http://localhost:8000',
      '/status':       'http://localhost:8000',
    }
  },
  define: {
    'process.env.VITE_API_URL': JSON.stringify(process.env.VITE_API_URL || '')
  }
})
