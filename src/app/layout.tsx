import type { Metadata } from "next";
import { Newsreader, JetBrains_Mono } from "next/font/google";
import { ClientOnly } from "@/components/ClientOnly";

import "./globals.css";

// Newsreader — the single editorial face of the product (body + headings).
// Optical sizing (opsz) lets the same family read as hairline display at large
// sizes and as sturdy text at body sizes. Oldstyle figures (`onum`) enabled in
// prose via `.academic-prose`.
const newsreader = Newsreader({
  variable: "--font-serif",
  subsets: ["latin"],
  // Variable font: full weight range available via font-weight. Using
  // `weight: "variable"` is required when specifying custom `axes` (opsz).
  weight: "variable",
  style: ["normal", "italic"],
  axes: ["opsz"],
  display: "swap",
});

// JetBrains Mono — apparatus only (folios, eyebrows, page counters, code).
// Never used for headings or body; mono-as-metadata is the distinction.
const jetbrainsMono = JetBrains_Mono({
  variable: "--font-mono-apparatus",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "TUTOR.AI — Learn smarter with documents",
  description:
    "Upload PDFs and chat with an AI tutor grounded in your documents, with page-accurate citations.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <meta
          name="viewport"
          content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no"
        />
        <link
          rel="stylesheet"
          href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css"
          integrity="sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV"
          crossOrigin="anonymous"
        />
        <link
          rel="stylesheet"
          href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css"
          integrity="sha512-DTOQO9RWCH3ppGqcWaEA1BIZOC6xxalwEsw9c2QQeAIftl+Vegovlnee1c9QX4TctnWMn13TZye+giMm8e2LwA=="
          crossOrigin="anonymous"
          referrerPolicy="no-referrer"
        />
      </head>
      <body
        className={`${newsreader.variable} ${jetbrainsMono.variable} antialiased`}
      >
        <ClientOnly>{children}</ClientOnly>
      </body>
    </html>
  );
}
