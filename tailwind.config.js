/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        slate: {
          900: '#1E293B',
        },
        electric: {
          blue: '#3B82F6',
        },
        lime: {
          green: '#22C55E',
        },
        soft: {
          yellow: '#FACC15',
        },
        light: {
          gray: '#F1F5F9',
        },
      },
    },
  },
  plugins: [],
}

