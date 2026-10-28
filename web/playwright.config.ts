import { defineConfig, devices } from '@playwright/test'

// Separate ports from `make dev` so the e2e run never touches a live session.
const API_PORT = 8417
const WEB_PORT = 5417
const uv = process.env.UV ?? 'uv'

export default defineConfig({
  testDir: './e2e',
  timeout: 240_000,
  fullyParallel: false,
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: `http://127.0.0.1:${WEB_PORT}`,
    ...devices['Desktop Chrome'],
    trace: 'retain-on-failure',
  },
  webServer: [
    {
      command: `${uv} run uvicorn --factory arena.api.app:create_app --host 127.0.0.1 --port ${API_PORT}`,
      cwd: '..',
      url: `http://127.0.0.1:${API_PORT}/health`,
      env: {
        ARENA_BOT_DELAY_MS: '40',
        ARENA_HAND_PAUSE_MS: '0',
        ARENA_ALLOWED_ORIGINS: `http://127.0.0.1:${WEB_PORT}`,
      },
      reuseExistingServer: false,
    },
    {
      command: 'npx vite',
      url: `http://127.0.0.1:${WEB_PORT}`,
      env: { ARENA_API_PORT: String(API_PORT), ARENA_WEB_PORT: String(WEB_PORT) },
      reuseExistingServer: false,
    },
  ],
})
