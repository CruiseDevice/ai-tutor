import type { Config } from "tailwindcss";
import typographyPlugin from '@tailwindcss/typography';
import containerQueries from '@tailwindcss/container-queries';

export default {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        // Newsreader (body + headings) and JetBrains Mono (apparatus only).
        // Both are loaded via next/font in src/app/layout.tsx.
        serif: ['var(--font-serif)', 'Georgia', 'Times New Roman', 'serif'],
        mono:  ['var(--font-mono-apparatus)', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      colors: {
        // Grounds
        paper:   'var(--paper)',
        surface: 'var(--surface)',
        desk:    'var(--desk)',
        'desk-2':'var(--desk-2)',

        // Ink
        ink:     'var(--ink)',
        'ink-2': 'var(--ink-2)',
        subtle:  'var(--subtle)',
        faint:   'var(--faint)',

        // Lines
        hair:        'var(--hair)',
        'hair-soft': 'var(--hair-soft)',

        // Accent (ink blue)
        accent:       'var(--accent)',
        'accent-ink': 'var(--accent-ink)',
        'accent-soft':'var(--accent-soft)',

        // Citation highlighter
        mark:      'var(--mark)',
        'mark-edge':'var(--mark-edge)',

        // Semantic
        danger:  'var(--danger)',
        success: 'var(--success)',
      },
      borderRadius: {
        'xs': 'var(--r-xs)',
        'sm': 'var(--r-sm)',
        DEFAULT: 'var(--r)',
        'lg': 'var(--r-lg)',
      },
      boxShadow: {
        'page': 'var(--shadow-page)',
        'card': 'var(--shadow-card)',
      },
      transitionTimingFunction: {
        'desk': 'var(--ease)',
      },
    },
  },
  plugins: [
    typographyPlugin,
    containerQueries,
  ],
} satisfies Config;
