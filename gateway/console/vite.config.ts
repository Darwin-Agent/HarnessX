import path from 'path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@lab': path.resolve(__dirname, '../../frontend/src'),
      '@gw':  path.resolve(__dirname, 'src'),
    },
  },
  server: {
    port: 8081,
    proxy: {
      '/gateway': 'http://localhost:8080',
      '/api':     'http://localhost:8080',
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    emptyOutDir: true,
  },
  base: '/console/',
})
