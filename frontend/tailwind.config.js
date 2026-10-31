/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0a0a0a',
        card: '#191919',
        border: '#212327',
        mule: '#f0596b',
        legit: '#4fd1a1',
        telemetry: '#c4b5fd',
        highlight: '#a0c3ec',
        warning: '#f5b14c',
      },
      fontFamily: {
        mono: ['Geist Mono', 'JetBrains Mono', 'ui-monospace', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
