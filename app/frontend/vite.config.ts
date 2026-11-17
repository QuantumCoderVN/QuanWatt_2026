import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        // Quapp-backed solves can run 10–15+ minutes with SSE progress
        timeout: 1_800_000,
        proxyTimeout: 1_800_000,
      }
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
  }
})
