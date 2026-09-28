import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    host: '127.0.0.1',
    watch: {
      ignored: ['**/.cache/**', '**/artifacts/**', '**/.devenv/**', '**/.direnv/**'],
    },
  },
  preview: { host: '127.0.0.1', port: 4177, strictPort: true },
});
