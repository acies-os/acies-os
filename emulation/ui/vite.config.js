import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';

// The monitor (emulation/monitor.py) serves the built app and its WebSocket on :8780.
// `npm run dev` serves the app with hot reload on :5174 and forwards /ws to that monitor.
const monitor = process.env.MONITOR_URL || 'http://127.0.0.1:8780';

export default defineConfig({
  plugins: [vue()],
  server: {
    host: '127.0.0.1',
    port: 5174,
    proxy: { '/ws': { target: monitor.replace(/^http/, 'ws'), ws: true } },
  },
  build: { outDir: 'dist', emptyOutDir: true },
});
