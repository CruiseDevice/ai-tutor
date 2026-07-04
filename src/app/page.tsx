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
    <div className="surface-card overflow-hidden">
      <div className="grid lg:grid-cols-2">
        {/* Left: a faux PDF page lifted off a desk-tinted ground */}
        <div className="p-5 sm:p-6 desk">
          <div className="bg-surface rounded shadow-page paper-grain p-5 sm:p-6 min-h-[260px]">
            <div className="flex items-center justify-between mb-4 pb-3 border-b border-hair-soft">
              <span className="font-serif text-sm text-ink-2 truncate">Research Paper.pdf</span>
              <span className="font-mono text-xs text-faint">3 / 15</span>
            </div>
            <p className="font-mono text-[0.65rem] uppercase tracking-widest text-faint mb-2">Abstract</p>
            <p className="font-serif text-sm leading-relaxed text-ink-2 mb-3">
              This paper analyzes machine learning algorithms and their application in predictive analytics.
            </p>
            <p className="font-serif text-sm leading-relaxed text-ink">
              <span key={highlightKey} className={highlightClass}>
                Our central hypothesis is that machine learning algorithms can significantly improve prediction accuracy
              </span>{" "}
              compared to traditional statistical methods.
            </p>
          </div>
        </div>

        {/* Right: the chat exchange */}
        <div className="p-5 sm:p-6 border-t lg:border-t-0 lg:border-l border-hair flex flex-col gap-4 justify-center">
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
          <div className="inline-block h-12 w-12 animate-spin rounded-full border-4 border-subtle border-t-ink border-r-accent"></div>
          <p className="mt-6 font-serif text-ink">Loading your learning experience...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="scroll-smooth bg-paper">
      {/* Navigation */}
      <nav className="fixed w-full z-50 bg-paper border-b-2 border-ink">
        <div className="flex justify-between items-center px-6 py-4">
          <div className="flex items-center gap-6">
            <Link href="/" className="font-mono text-xl font-bold tracking-tight">
              STUDYFETCH<span className="text-accent">.</span>AI
            </Link>
            <div className="hidden md:flex items-center gap-1 font-mono text-xs">
              <span className="text-subtle">[</span>
              <a href="#features" className="px-3 py-2 hover:text-accent transition-colors">FEATURES</a>
              <span className="text-subtle">]</span>
              <span className="text-subtle">[</span>
              <a href="#how-it-works" className="px-3 py-2 hover:text-accent transition-colors">HOW</a>
              <span className="text-subtle">]</span>
              <span className="text-subtle">[</span>
              <a href="#demo" className="px-3 py-2 hover:text-accent transition-colors">DEMO</a>
              <span className="text-subtle">]</span>
            </div>
          </div>
          <div className="flex items-center gap-4 font-mono text-xs">
            <a href="/login" className="px-4 py-2 hover:text-accent transition-colors">LOGIN</a>
            <a href="/register" className="bg-ink text-paper px-6 py-3 border-2 border-ink hover:bg-accent hover:border-accent transition-colors">
              GET STARTED
            </a>
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
              <a href="/register" className="btn btn-primary">Start free</a>
              <a href="#how-it-works" className="btn">See how it works</a>
            </div>
          </div>

          {/* Right: the looping citation moment */}
          <div className="lg:col-span-5">
            <HeroDemo />
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-24 bg-paper border-b-2 border-ink">
        <div className="px-6 md:px-12">
          {/* Section header */}
          <div className="flex flex-col md:flex-row md:items-end md:justify-between mb-16 pb-8 border-b border-ink">
            <div>
              <span className="font-mono text-xs text-accent">[002]</span>
              <h2 className="font-mono text-4xl md:text-6xl font-bold mt-2">
                FEATURES
              </h2>
            </div>
            <p className="font-serif text-lg text-subtle max-w-md mt-4 md:mt-0">
              Everything you need to transform your study materials into an interactive learning experience.
            </p>
          </div>

          {/* Features grid - asymmetric layout */}
          <div className="grid md:grid-cols-2 gap-px bg-ink border-2 border-ink">
            {/* Feature 1 - Full width on desktop */}
            <div className="bg-paper p-8 md:p-12 hover:bg-accent/5 transition-colors group">
              <div className="flex items-start gap-6">
                <span className="font-mono text-6xl font-bold text-accent/20 group-hover:text-accent/40">01</span>
                <div>
                  <h3 className="font-mono text-xl font-bold mb-2">PDF Upload & Processing</h3>
                  <p className="font-serif text-subtle">Upload your PDF documents. Our AI instantly processes them, making every page searchable and interactive.</p>
                </div>
              </div>
            </div>

            {/* Feature 2 */}
            <div className="bg-paper p-8 md:p-12 hover:bg-accent/5 transition-colors group">
              <div className="flex items-start gap-6">
                <span className="font-mono text-6xl font-bold text-accent/20 group-hover:text-accent/40">02</span>
                <div>
                  <h3 className="font-mono text-xl font-bold mb-2">Intelligent Chat Interface</h3>
                  <p className="font-serif text-subtle">Ask questions naturally. Get detailed explanations with exact page references.</p>
                </div>
              </div>
            </div>

            {/* Feature 3 */}
            <div className="bg-paper p-8 md:p-12 hover:bg-accent/5 transition-colors group">
              <div className="flex items-start gap-6">
                <span className="font-mono text-6xl font-bold text-accent/20 group-hover:text-accent/40">03</span>
                <div>
                  <h3 className="font-mono text-xl font-bold mb-2">Smart Document Search</h3>
                  <p className="font-serif text-subtle">Advanced vector search finds relevant content even when you don&apos;t remember exact keywords.</p>
                </div>
              </div>
            </div>

            {/* Feature 4 */}
            <div className="bg-paper p-8 md:p-12 hover:bg-accent/5 transition-colors group">
              <div className="flex items-start gap-6">
                <span className="font-mono text-6xl font-bold text-accent/20 group-hover:text-accent/40">04</span>
                <div>
                  <h3 className="font-mono text-xl font-bold mb-2">Persistent Conversations</h3>
                  <p className="font-serif text-subtle">Chat history is automatically saved. Pick up where you left off.</p>
                </div>
              </div>
            </div>

            {/* Feature 5 */}
            <div className="bg-paper p-8 md:p-12 hover:bg-accent/5 transition-colors group">
              <div className="flex items-start gap-6">
                <span className="font-mono text-6xl font-bold text-accent/20 group-hover:text-accent/40">05</span>
                <div>
                  <h3 className="font-mono text-xl font-bold mb-2">Multi-Document Support</h3>
                  <p className="font-serif text-subtle">Manage multiple documents with separate conversation histories for each subject.</p>
                </div>
              </div>
            </div>

            {/* Feature 6 */}
            <div className="bg-paper p-8 md:p-12 hover:bg-accent/5 transition-colors group">
              <div className="flex items-start gap-6">
                <span className="font-mono text-6xl font-bold text-accent/20 group-hover:text-accent/40">06</span>
                <div>
                  <h3 className="font-mono text-xl font-bold mb-2">Secure & Private</h3>
                  <p className="font-serif text-subtle">Your documents are encrypted. Only you have access to your data.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="py-24 bg-paper border-b-2 border-ink">
        <div className="px-6 md:px-12">
          <div className="mb-16 pb-8 border-b border-ink">
            <span className="font-mono text-xs text-accent">[003]</span>
            <h2 className="font-mono text-4xl md:text-6xl font-bold mt-2">HOW IT WORKS</h2>
          </div>

          <div className="flex flex-col md:flex-row">
            {/* Step 1 */}
            <div className="flex-1 p-8 border-b-2 md:border-b-0 md:border-r border-ink">
              <span className="font-mono text-8xl font-bold text-accent/20">01</span>
              <h3 className="font-mono text-xl font-bold mt-6 mb-4">UPLOAD YOUR PDF</h3>
              <p className="font-serif text-subtle">Drag and drop your study materials, textbooks, or research papers. Our AI processes them instantly.</p>
            </div>

            {/* Step 2 */}
            <div className="flex-1 p-8 border-b-2 md:border-b-0 md:border-r border-ink">
              <span className="font-mono text-8xl font-bold text-accent/20">02</span>
              <h3 className="font-mono text-xl font-bold mt-6 mb-4">ASK QUESTIONS</h3>
              <p className="font-serif text-subtle">Start chatting with your document. Ask for explanations, summaries, or specific information.</p>
            </div>

            {/* Step 3 */}
            <div className="flex-1 p-8">
              <span className="font-mono text-8xl font-bold text-accent/20">03</span>
              <h3 className="font-mono text-xl font-bold mt-6 mb-4">GET SMART ANSWERS</h3>
              <p className="font-serif text-subtle">Receive detailed, contextual answers with page references. Learn faster.</p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 bg-ink text-paper border-t-2 border-ink">
        <div className="px-6 md:px-12 max-w-5xl">
          <span className="font-mono text-xs text-accent">[005]</span>
          <h2 className="font-mono text-4xl md:text-6xl font-bold mt-2 mb-8">
            READY TO TRANSFORM<br/>YOUR LEARNING?
          </h2>
          <p className="font-serif text-xl text-subtle mb-12 max-w-2xl">
            Join thousands of students learning smarter with AI-powered tutoring.
          </p>
          <div className="flex flex-col sm:flex-row gap-4">
            <a href="/register" className="font-mono text-sm bg-paper text-ink px-10 py-5 border-3 border-paper hover:bg-accent hover:border-accent hover:text-paper transition-colors text-center">
              GET STARTED FREE
            </a>
            <a href="/login" className="font-mono text-sm bg-transparent text-paper px-10 py-5 border-2 border-subtle hover:border-paper transition-colors text-center">
              SIGN IN
            </a>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-paper py-12 border-t-2 border-ink">
        <div className="px-6 md:px-12">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-8">
            <div>
              <span className="font-mono text-xl font-bold">STUDYFETCH.AI</span>
              <p className="font-serif text-sm text-subtle mt-2">&copy; 2024 All rights reserved</p>
            </div>
            <div className="flex gap-8 font-mono text-xs">
              <a href="#" className="hover:text-accent transition-colors">PRIVACY</a>
              <a href="#" className="hover:text-accent transition-colors">TERMS</a>
              <a href="#" className="hover:text-accent transition-colors">HELP</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
