import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const apiPort = process.env.ARENA_API_PORT ?? '8317'
const webPort = Number(process.env.ARENA_WEB_PORT ?? '5317')

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: webPort,
    strictPort: true,
    proxy: {
      '/api': {
        target: `http://127.0.0.1:${apiPort}`,
        ws: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
