import type { Config } from "tailwindcss";
import typographyPlugin from '@tailwindcss/typography';

export default {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        mono: ['var(--font-mono)', 'monospace'],
        serif: ['var(--font-serif)', 'Georgia', 'serif'],
      },
      colors: {
        // Legacy (for compatibility)
        background: "var(--background)",
        foreground: "var(--foreground)",
        // Brutalist palette
        ink: 'var(--ink)',
        paper: 'var(--paper)',
        accent: 'var(--accent)',
        rule: 'var(--rule)',
        subtle: 'var(--subtle)',
      },
      borderWidth: {
        '3': '3px',
      },
    },
  },
  plugins: [
    typographyPlugin,
  ],
} satisfies Config;
