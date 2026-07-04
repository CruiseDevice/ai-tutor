// app/page.tsx
"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import Link from "next/link";
import { authApi } from "@/lib/api-client";

// ───────────────────────────────────────────────────────────────
// HeroDemo — a calm, looping citation-jump moment.
//
// CSS-transition-driven (no per-character typing): fewer timers,
// smoother, and trivial to freeze. Under prefers-reduced-motion it
// holds the final frame (answer visible, passage marked) — no loop.
// Reuses only Quiet Desk utilities shipped in Phases 0–2.
// ───────────────────────────────────────────────────────────────

const HERO = {
  IDLE: 0,
  QUESTION: 1,
  ANSWER: 2,
  HIGHLIGHT: 3,
  HOLD: 4,
} as const;

function HeroDemo() {
  const [phase, setPhase] = useState<number>(HERO.IDLE);
  const [reduced, setReduced] = useState(false);
  const [highlightKey, setHighlightKey] = useState(0);

  // Detect reduced-motion once on mount (avoids SSR/CSR mismatch).
  useEffect(() => {
    setReduced(window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  }, []);

  // Single timeline-driven loop. When reduced, park on the final frame.
  useEffect(() => {
    if (reduced) {
      setPhase(HERO.HIGHLIGHT);
      return;
    }
    let mounted = true;
    const timers: ReturnType<typeof setTimeout>[] = [];
    const run = () => {
      setPhase(HERO.IDLE);
      timers.push(setTimeout(() => mounted && setPhase(HERO.QUESTION), 400));
      timers.push(setTimeout(() => mounted && setPhase(HERO.ANSWER), 1800));
      timers.push(
        setTimeout(() => {
          if (!mounted) return;
          setPhase(HERO.HIGHLIGHT);
          // Bump the key so the mark-fade animation replays each loop.
          setHighlightKey((n) => n + 1);
        }, 2750)
      );
      timers.push(setTimeout(() => mounted && setPhase(HERO.HOLD), 5200));
      timers.push(setTimeout(() => mounted && run(), 6800));
    };
    run();
    return () => {
      mounted = false;
      timers.forEach(clearTimeout);
    };
  }, [reduced]);

  const marked = phase >= HERO.HIGHLIGHT;
  // Reduced-motion gets a static mark; otherwise the 2.4s mark-fade sweep.
  const highlightClass = marked
    ? reduced
      ? "bg-mark rounded-[2px] px-0.5"
      : "mark-fade rounded-[2px] px-0.5"
    : "";

  return (
    // The demo keys its internal layout off its OWN width (@container), not the
    // viewport. In the desktop hero it lives in a narrow col-span-5 strip and
    // stacks; only when the card itself is wide enough does it go side-by-side.
    <div className="surface-card overflow-hidden @container">
      <div className="grid grid-cols-1 @[36rem]:grid-cols-2">
        {/* Left: a faux PDF page lifted off a desk-tinted ground */}
        <div className="p-4 @[36rem]:p-5 desk">
          <div className="bg-surface rounded shadow-page paper-grain p-4 @[36rem]:p-5 min-h-[220px] @[36rem]:min-h-[260px]">
            <div className="flex items-center justify-between mb-3 pb-2 @[36rem]:mb-4 @[36rem]:pb-3 border-b border-hair-soft">
              <span className="font-serif text-xs @[36rem]:text-sm text-ink-2 truncate">Research Paper.pdf</span>
              <span className="font-mono text-xs text-faint">3 / 15</span>
            </div>
            <p className="font-mono text-[0.65rem] uppercase tracking-widest text-faint mb-2">Abstract</p>
            <p className="font-serif text-xs @[36rem]:text-sm leading-relaxed text-ink-2 mb-3">
              This paper analyzes machine learning algorithms and their application in predictive analytics.
            </p>
            <p className="font-serif text-xs @[36rem]:text-sm leading-relaxed text-ink">
              <span key={highlightKey} className={highlightClass}>
                Our central hypothesis is that machine learning algorithms can significantly improve prediction accuracy
              </span>{" "}
              compared to traditional statistical methods.
            </p>
          </div>
        </div>

        {/* Right: the chat exchange */}
        <div className="p-4 @[36rem]:p-5 border-t @[36rem]:border-t-0 @[36rem]:border-l border-hair flex flex-col gap-4 justify-center">
          {/* User question */}
          <div
            className={`flex justify-end transition-opacity duration-500 ${
              phase >= HERO.QUESTION ? "opacity-100" : "opacity-0"
            }`}
          >
            <div className="desk rounded-[10px] rounded-tr-sm px-4 py-2.5 max-w-[85%]">
              <p className="font-serif text-sm text-ink">What&apos;s the main hypothesis?</p>
            </div>
          </div>

          {/* AI answer with the citation signature */}
          <div
            className={`flex gap-3 transition-opacity duration-700 ${
              phase >= HERO.ANSWER ? "opacity-100" : "opacity-0"
            }`}
          >
            <div className="tutor-avatar">A</div>
            <div className="font-serif text-sm leading-relaxed text-ink flex-1 pt-1">
              The paper argues that{" "}
              <em className="text-ink-2">
                machine learning algorithms can significantly improve prediction accuracy
              </em>
              <sup className="citation">3</sup> over traditional statistical methods.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function Home() {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // check if user is authenticated
    const checkAuth = async () => {
      try {
        const response = await authApi.verifySession();

        // if authenticated, redirect to dashboard
        if(response.ok) {
          router.push('/dashboard')
        } else {
          // if not authenticated, show landing page
          setIsLoading(false);
        }
      } catch (error) {
        // on error, show landing page
        setIsLoading(false);
        console.error('Auth check error: ', error);
      }
    };
    checkAuth();
  }, [router]);

  if (isLoading) {
    return (
      <div className="h-screen flex items-center justify-center bg-paper">
        <div className="text-center">
          <div className="inline-block h-10 w-10 animate-spin rounded-full border-2 border-hair border-t-accent"></div>
          <p className="mt-6 font-serif text-faint">Loading…</p>
        </div>
      </div>
    );
  }

  return (
    <div className="scroll-smooth bg-paper">
      {/* Navigation — paper, hairline, TUTOR.AI wordmark */}
      <nav className="fixed w-full z-50 bg-paper border-b border-hair">
        <div className="flex justify-between items-center px-6 py-4">
          <div className="flex items-center gap-8">
            <Link href="/" className="font-serif text-xl font-semibold tracking-tight text-ink">
              TUTOR<span className="text-accent">.</span>AI
            </Link>
            <div className="hidden md:flex items-center gap-6 font-serif text-sm">
              <a href="#features" className="text-subtle hover:text-accent transition-colors">Features</a>
              <a href="#how-it-works" className="text-subtle hover:text-accent transition-colors">How it works</a>
              <a href="#cta" className="text-subtle hover:text-accent transition-colors">Get started</a>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <Link href="/login" className="font-serif text-sm text-subtle hover:text-accent transition-colors px-3 py-2">
              Log in
            </Link>
            <Link href="/register" className="btn btn-primary text-sm">
              Get started
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero Section — the product's magic, shown not described */}
      <section className="pt-32 pb-16 lg:py-28 bg-paper">
        <div className="max-w-7xl mx-auto px-6 lg:px-8 grid lg:grid-cols-12 gap-12 lg:gap-16 items-center">
          {/* Left: copy */}
          <div className="lg:col-span-7 text-center lg:text-left">
            <p className="font-mono text-xs uppercase tracking-widest text-faint mb-6">
              AI tutoring, grounded in your PDF
            </p>
            <h1 className="font-serif text-5xl md:text-6xl lg:text-7xl font-semibold leading-[1.05] tracking-tight text-ink mb-6">
              Ask your PDF{" "}
              <span className="text-accent italic">anything.</span>
            </h1>
            <p className="font-serif text-lg md:text-xl text-subtle leading-relaxed max-w-xl mx-auto lg:mx-0 mb-10">
              Every answer shows its source — click a citation and jump straight to the passage in the document.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start">
              <a href="/register" className="btn btn-primary w-full sm:w-auto">Start free</a>
              <a href="#how-it-works" className="btn w-full sm:w-auto">See how it works</a>
            </div>
          </div>

          {/* Right: the looping citation moment */}
          <div className="lg:col-span-5">
            <HeroDemo />
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-24 bg-paper border-t border-hair">
        <div className="px-6 md:px-12">
          {/* Section header */}
          <div className="flex flex-col md:flex-row md:items-end md:justify-between mb-16 pb-8 border-b border-hair">
            <div>
              <p className="font-mono text-xs uppercase tracking-widest text-faint mb-2">What you get</p>
              <h2 className="font-serif text-4xl md:text-5xl font-semibold text-ink">
                Features
              </h2>
            </div>
            <p className="font-serif text-lg text-subtle max-w-md mt-4 md:mt-0">
              Everything you need to transform your study materials into an interactive learning experience.
            </p>
          </div>

          {/* Features grid — hairline grid, subtle apparatus numerals */}
          <div className="grid md:grid-cols-2 gap-px bg-hair border border-hair rounded overflow-hidden">
            {/* Feature 1 */}
            <div className="bg-surface p-8 md:p-10 hover:bg-accent-soft/40 transition-colors">
              <div className="flex items-start gap-5">
                <span className="font-mono text-sm text-faint pt-1">01</span>
                <div>
                  <h3 className="font-serif text-lg font-semibold text-ink mb-2">PDF upload & processing</h3>
                  <p className="font-serif text-subtle leading-relaxed">Upload your PDF documents. Our AI instantly processes them, making every page searchable and interactive.</p>
                </div>
              </div>
            </div>

            {/* Feature 2 */}
            <div className="bg-surface p-8 md:p-10 hover:bg-accent-soft/40 transition-colors">
              <div className="flex items-start gap-5">
                <span className="font-mono text-sm text-faint pt-1">02</span>
                <div>
                  <h3 className="font-serif text-lg font-semibold text-ink mb-2">Intelligent chat interface</h3>
                  <p className="font-serif text-subtle leading-relaxed">Ask questions naturally. Get detailed explanations with exact page references.</p>
                </div>
              </div>
            </div>

            {/* Feature 3 */}
            <div className="bg-surface p-8 md:p-10 hover:bg-accent-soft/40 transition-colors">
              <div className="flex items-start gap-5">
                <span className="font-mono text-sm text-faint pt-1">03</span>
                <div>
                  <h3 className="font-serif text-lg font-semibold text-ink mb-2">Smart document search</h3>
                  <p className="font-serif text-subtle leading-relaxed">Advanced vector search finds relevant content even when you don&apos;t remember exact keywords.</p>
                </div>
              </div>
            </div>

            {/* Feature 4 */}
            <div className="bg-surface p-8 md:p-10 hover:bg-accent-soft/40 transition-colors">
              <div className="flex items-start gap-5">
                <span className="font-mono text-sm text-faint pt-1">04</span>
                <div>
                  <h3 className="font-serif text-lg font-semibold text-ink mb-2">Persistent conversations</h3>
                  <p className="font-serif text-subtle leading-relaxed">Chat history is automatically saved. Pick up where you left off.</p>
                </div>
              </div>
            </div>

            {/* Feature 5 */}
            <div className="bg-surface p-8 md:p-10 hover:bg-accent-soft/40 transition-colors">
              <div className="flex items-start gap-5">
                <span className="font-mono text-sm text-faint pt-1">05</span>
                <div>
                  <h3 className="font-serif text-lg font-semibold text-ink mb-2">Multi-document support</h3>
                  <p className="font-serif text-subtle leading-relaxed">Manage multiple documents with separate conversation histories for each subject.</p>
                </div>
              </div>
            </div>

            {/* Feature 6 */}
            <div className="bg-surface p-8 md:p-10 hover:bg-accent-soft/40 transition-colors">
              <div className="flex items-start gap-5">
                <span className="font-mono text-sm text-faint pt-1">06</span>
                <div>
                  <h3 className="font-serif text-lg font-semibold text-ink mb-2">Secure & private</h3>
                  <p className="font-serif text-subtle leading-relaxed">Your documents are encrypted. Only you have access to your data.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="py-24 bg-paper border-t border-hair">
        <div className="px-6 md:px-12">
          <div className="mb-16 pb-8 border-b border-hair">
            <p className="font-mono text-xs uppercase tracking-widest text-faint mb-2">A simple sequence</p>
            <h2 className="font-serif text-4xl md:text-5xl font-semibold text-ink">How it works</h2>
          </div>

          <div className="flex flex-col md:flex-row">
            {/* Step 1 */}
            <div className="flex-1 p-8 border-b md:border-b-0 md:border-r border-hair">
              <span className="font-mono text-sm text-accent">01</span>
              <h3 className="font-serif text-lg font-semibold text-ink mt-4 mb-3">Upload your PDF</h3>
              <p className="font-serif text-subtle leading-relaxed">Drag and drop your study materials, textbooks, or research papers. Our AI processes them instantly.</p>
            </div>

            {/* Step 2 */}
            <div className="flex-1 p-8 border-b md:border-b-0 md:border-r border-hair">
              <span className="font-mono text-sm text-accent">02</span>
              <h3 className="font-serif text-lg font-semibold text-ink mt-4 mb-3">Ask questions</h3>
              <p className="font-serif text-subtle leading-relaxed">Start chatting with your document. Ask for explanations, summaries, or specific information.</p>
            </div>

            {/* Step 3 */}
            <div className="flex-1 p-8">
              <span className="font-mono text-sm text-accent">03</span>
              <h3 className="font-serif text-lg font-semibold text-ink mt-4 mb-3">Get grounded answers</h3>
              <p className="font-serif text-subtle leading-relaxed">Receive detailed, contextual answers with page references. Learn faster.</p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section — paper throughout, hairline divider (not inverted ink) */}
      <section id="cta" className="py-24 bg-paper border-t border-hair">
        <div className="px-6 md:px-12 max-w-5xl">
          <h2 className="font-serif text-4xl md:text-5xl font-semibold text-ink mb-6 leading-tight">
            Ready to read deeper?
          </h2>
          <p className="font-serif text-lg md:text-xl text-subtle mb-12 max-w-2xl leading-relaxed">
            Upload a PDF and ask your first question in under a minute. Every answer cites its source.
          </p>
          <div className="flex flex-col sm:flex-row gap-4">
            <a href="/register" className="btn btn-primary">Start free</a>
            <a href="/login" className="btn">Sign in</a>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-paper py-12 border-t border-hair">
        <div className="px-6 md:px-12">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-8">
            <div>
              <span className="font-serif text-xl font-semibold text-ink">TUTOR<span className="text-accent">.</span>AI</span>
              <p className="font-serif text-sm text-subtle mt-2">&copy; {new Date().getFullYear()} All rights reserved</p>
            </div>
            <div className="flex gap-8 font-serif text-sm">
              <a href="#" className="text-subtle hover:text-accent transition-colors">Privacy</a>
              <a href="#" className="text-subtle hover:text-accent transition-colors">Terms</a>
              <a href="#" className="text-subtle hover:text-accent transition-colors">Help</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
