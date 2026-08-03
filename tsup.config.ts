import { defineConfig } from 'tsup';

export default defineConfig({
  entry: ['index.ts'],
  format: ['cjs', 'esm'], // host loads the CommonJS build via plugin.json "main"
  dts: true,
  sourcemap: true,
  clean: true,
  // Provided by the host at runtime — never bundle these into the plugin.
  external: ['react', 'react-dom', '@signalsandsorcery/plugin-sdk'],
  treeshake: true,
  // bundle the shipped morph graph into dist
  loader: { '.json': 'json' },
});
