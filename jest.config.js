/** @type {import('jest').Config} */
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'jsdom',
  testMatch: ['**/__tests__/**/*.test.ts', '**/__tests__/**/*.test.tsx'],
  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json'],
  transform: {
    '^.+\\.tsx?$': ['ts-jest', { tsconfig: { jsx: 'react-jsx', resolveJsonModule: true } }],
  },
  // Resolve the SDK against the locally-installed (file:) copy.
  moduleNameMapper: {
    '^@signalsandsorcery/plugin-sdk$': '<rootDir>/node_modules/@signalsandsorcery/plugin-sdk',
    // The SDK is a file: link with its own react copy; two Reacts in one
    // renderer breaks hooks ("Cannot read properties of null (reading useRef)").
    '^react$': '<rootDir>/node_modules/react',
    '^react-dom$': '<rootDir>/node_modules/react-dom',
    '^react-dom/(.*)$': '<rootDir>/node_modules/react-dom/$1',
    '^react/jsx-runtime$': '<rootDir>/node_modules/react/jsx-runtime',
  },
  testPathIgnorePatterns: ['/node_modules/', '/dist/', '/training/'],
};
