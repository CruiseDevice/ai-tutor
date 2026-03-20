// app/page.tsx
"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState, useCallback, useRef } from "react";
import { authApi } from "@/lib/api-client";

// Animated Demo Component
const DEMO_MESSAGES = [
  "Can you explain the main hypothesis in this paper?",
  "Based on [Page 3], the main hypothesis is that machine learning algorithms can significantly improve prediction accuracy...",
  "What evidence supports this?",
];

function AnimatedDemo() {
  const [stage, setStage] = useState(0);
  const [typingText, setTypingText] = useState("");
  const [showMessages, setShowMessages] = useState({
    message1: false,
    message2: false,
    message3: false,
  });
  const [highlightedLine, setHighlightedLine] = useState<number | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [isTyping, setIsTyping] = useState(false);
  const [isAIThinking, setIsAIThinking] = useState(false);
  const [aiResponseText, setAIResponseText] = useState("");
  const typingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const aiTypingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const documentRef = useRef<HTMLDivElement>(null);
  const abstractRef = useRef<HTMLDivElement>(null);

  const typeMessage = useCallback((text: string, onComplete: () => void) => {
    // Clear any existing interval
    if (typingIntervalRef.current) {
      clearInterval(typingIntervalRef.current);
    }

    let index = 0;
    setTypingText("");
    typingIntervalRef.current = setInterval(() => {
      if (index < text.length) {
        setTypingText(text.slice(0, index + 1));
        index++;
      } else {
        if (typingIntervalRef.current) {
          clearInterval(typingIntervalRef.current);
          typingIntervalRef.current = null;
        }
        onComplete();
      }
    }, 30);
  }, []);

  const typeAIResponse = useCallback((text: string, onComplete: () => void) => {
    // Clear any existing interval
    if (aiTypingIntervalRef.current) {
      clearInterval(aiTypingIntervalRef.current);
    }

    let index = 0;
    setAIResponseText("");
    aiTypingIntervalRef.current = setInterval(() => {
      if (index < text.length) {
        setAIResponseText(text.slice(0, index + 1));
        index++;
      } else {
        if (aiTypingIntervalRef.current) {
          clearInterval(aiTypingIntervalRef.current);
          aiTypingIntervalRef.current = null;
        }
        onComplete();
      }
    }, 20);
  }, []);

  useEffect(() => {
    let timer: NodeJS.Timeout;

    const runStage = async () => {
      switch (stage) {
        case 0: // User types first message
          timer = setTimeout(() => {
            setIsTyping(true);
            setTypingText("");
            typeMessage(DEMO_MESSAGES[0], () => {
              setIsTyping(false);
              setTypingText("");
              setShowMessages(prev => ({ ...prev, message1: true }));
              setStage(1);
            });
          }, 1000);
          break;

        case 1: // AI thinks
          timer = setTimeout(() => {
            setIsAIThinking(true);
            // Wait for thinking animation
            setTimeout(() => {
              setStage(2);
            }, 2000);
          }, 500);
          break;

        case 2: // AI responds
          setIsAIThinking(false);
          setHighlightedLine(2);
          setCurrentPage(3);

          setTimeout(() => {
            // Scroll within the document container, not the entire page
            if (abstractRef.current && documentRef.current) {
              const container = documentRef.current;
              const element = abstractRef.current;
              const containerRect = container.getBoundingClientRect();
              const elementRect = element.getBoundingClientRect();
              const relativeTop = elementRect.top - containerRect.top + container.scrollTop;
              const scrollPosition = relativeTop - (container.clientHeight / 2) + (element.clientHeight / 2);

              container.scrollTo({
                top: scrollPosition,
                behavior: 'smooth'
              });
            }
          }, 100);

          timer = setTimeout(() => {
            typeAIResponse(DEMO_MESSAGES[1], () => {
              setShowMessages(prev => ({ ...prev, message2: true }));
              setStage(3);
            });
          }, 400);
          break;

        case 3: // User types second message
          timer = setTimeout(() => {
            setIsTyping(true);
            setTypingText("");
            typeMessage(DEMO_MESSAGES[2], () => {
              setIsTyping(false);
              setTypingText("");
              setShowMessages(prev => ({ ...prev, message3: true }));
              setStage(4);
            });
          }, 1500);
          break;

        case 4: // Reset
          timer = setTimeout(() => {
            setShowMessages({ message1: false, message2: false, message3: false });
            setTypingText("");
            setAIResponseText("");
            setHighlightedLine(null);
            setCurrentPage(1);
            setIsTyping(false);
            setIsAIThinking(false);
            setTimeout(() => setStage(0), 500);
          }, 4000);
          break;
      }
    };

    runStage();

    return () => {
      clearTimeout(timer);
      if (typingIntervalRef.current) {
        clearInterval(typingIntervalRef.current);
        typingIntervalRef.current = null;
      }
      if (aiTypingIntervalRef.current) {
        clearInterval(aiTypingIntervalRef.current);
        aiTypingIntervalRef.current = null;
      }
    };
  }, [stage, typeMessage, typeAIResponse]);

  return (
    <section id="demo" className="py-24 bg-paper border-b-2 border-ink">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-20">
          <div className="font-mono text-xs text-accent mb-6">
            [004] INTERACTIVE DEMO
          </div>
          <h2 className="font-mono text-4xl md:text-6xl font-bold mb-6">
            SEE IT IN ACTION
          </h2>
          <p className="font-serif text-xl text-subtle max-w-3xl mx-auto leading-relaxed">
            Watch how StudyFetch transforms your learning experience
          </p>
        </div>

        <div className="max-w-5xl mx-auto">
          <div className="border-3 border-ink bg-paper">
            <div className="border-b-2 border-ink bg-ink text-paper px-6 py-3 font-mono text-xs">
              ▶ LIVE DEMO
            </div>
            <div className="p-8">
              <div className="flex flex-col lg:flex-row gap-6">
                {/* PDF Viewer Mockup */}
                <div ref={documentRef} className="lg:w-1/2 bg-paper border-2 border-ink p-6 overflow-y-auto max-h-[600px] relative">
                  <div className="flex items-center justify-between mb-4 pb-3 border-b border-ink sticky top-0 bg-paper z-10">
                    <div className="flex items-center gap-2">
                      <div className="w-8 h-8 bg-accent flex items-center justify-center font-mono text-xs font-bold text-paper">
                        PDF
                      </div>
                      <span className="text-sm font-mono font-semibold">Research Paper.pdf</span>
                    </div>
                    <span className="text-xs font-mono bg-subtle/20 px-2 py-1 transition-all duration-500">
                      Page {currentPage} / 15
                    </span>
                  </div>
                  <div className="prose prose-sm max-w-none leading-relaxed">
                    {/* Title */}
                    <h1 className={`text-xl font-bold mb-2 transition-all duration-500 ${highlightedLine === 0 ? 'bg-accent/20 px-2 py-1 border-l-2 border-accent' : ''}`}>
                      Machine Learning Algorithms for Predictive Analytics in Healthcare
                    </h1>

                    {/* Authors */}
                    <div className={`text-xs text-subtle mb-4 italic transition-all duration-500 ${highlightedLine === 1 ? 'bg-accent/20 px-2 py-1' : ''}`}>
                      John A. Smith, Sarah B. Johnson, Michael C. Williams
                    </div>

                    {/* Abstract */}
                    <div ref={abstractRef} className={`mb-4 transition-all duration-500 ${highlightedLine === 2 ? 'bg-accent/30 px-3 py-2 border-l-4 border-accent' : ''}`}>
                      <h2 className="text-sm font-mono font-semibold mb-2">Abstract</h2>
                      <p className={`text-xs leading-relaxed mb-2 transition-all duration-500 ${highlightedLine === 2 ? 'font-medium' : ''}`}>
                        This paper presents a comprehensive analysis of machine learning algorithms and their application in healthcare predictive analytics. We evaluate the performance of various models including neural networks, random forests, and support vector machines.
                      </p>
                      <p className={`text-xs leading-relaxed transition-all duration-500 ${highlightedLine === 3 ? 'bg-accent/20 px-2 py-1' : ''}`}>
                        Our results demonstrate that <span className={`transition-all duration-500 ${highlightedLine === 2 ? 'bg-accent font-semibold px-1.5 py-0.5' : ''}`}>machine learning algorithms can significantly improve prediction accuracy</span> compared to traditional statistical methods, with an average improvement of 23.4% across all tested datasets.
                      </p>
                    </div>

                    {/* Introduction */}
                    <div className={`mb-4 transition-all duration-500 ${highlightedLine === 4 ? 'bg-accent/20 px-3 py-2' : ''}`}>
                      <h2 className="text-sm font-mono font-semibold mb-2">1. Introduction</h2>
                      <p className="text-xs leading-relaxed mb-2">
                        The integration of artificial intelligence and machine learning in healthcare has revolutionized the way we approach medical diagnosis and treatment planning. Recent advances in computational power and data availability have enabled researchers to develop sophisticated predictive models.
                      </p>
                      <p className={`text-xs leading-relaxed transition-all duration-500 ${highlightedLine === 5 ? 'bg-accent/20 px-2 py-1' : ''}`}>
                        The primary objective of this research is to investigate the effectiveness of different machine learning approaches in predicting patient outcomes and identifying high-risk cases early in the treatment process.
                      </p>
                    </div>

                    {/* Methodology Section */}
                    <div className={`mb-4 transition-all duration-500 ${highlightedLine === 6 ? 'bg-accent/20 px-3 py-2' : ''}`}>
                      <h2 className="text-sm font-mono font-semibold mb-2">2. Methodology</h2>
                      <p className="text-xs leading-relaxed mb-2">
                        We conducted a systematic review of existing literature and performed empirical analysis on three distinct healthcare datasets. Each dataset contained anonymized patient records spanning a period of five years.
                      </p>
                      <p className={`text-xs leading-relaxed transition-all duration-500 ${highlightedLine === 7 ? 'bg-accent/30 px-2 py-1 border-l-2 border-accent' : ''}`}>
                        The experimental setup involved training multiple models using cross-validation techniques to ensure robust performance metrics. We compared accuracy, precision, recall, and F1-scores across all models.
                      </p>
                    </div>

                    {/* Results Preview */}
                    <div className={`mb-2 transition-all duration-500 ${highlightedLine === 8 ? 'bg-accent/20 px-3 py-2' : ''}`}>
                      <h2 className="text-sm font-mono font-semibold mb-2">3. Results</h2>
                      <p className="text-xs leading-relaxed">
                        Our analysis reveals significant improvements in prediction accuracy when using ensemble methods. The random forest classifier achieved the highest performance with 94.2% accuracy...
                      </p>
                    </div>
                  </div>
                </div>

                {/* Chat Interface Mockup */}
                <div className="lg:w-1/2 bg-paper border-2 border-ink p-6 flex flex-col max-h-[600px]">
                  <div className="flex items-center gap-2 border-b-2 border-ink pb-3 mb-4 flex-shrink-0">
                    <div className="w-8 h-8 bg-ink text-paper flex items-center justify-center font-mono text-xs font-bold">
                      AI
                    </div>
                    <span className="text-sm font-mono font-semibold">AI TUTOR CHAT</span>
                  </div>
                  <div className="space-y-4 flex-1 overflow-y-auto mb-4 min-h-0">
                    {/* First User Message */}
                    {showMessages.message1 && (
                      <div className="flex justify-end animate-slide-in-right">
                        <div className="bg-ink text-paper px-4 py-2.5 text-sm max-w-xs border-2 border-ink">
                          {DEMO_MESSAGES[0]}
                        </div>
                      </div>
                    )}

                    {/* AI Thinking Indicator */}
                    {isAIThinking && (
                      <div className="flex justify-start animate-slide-in-left">
                        <div className="bg-paper text-ink px-4 py-2.5 text-sm max-w-xs border-2 border-ink">
                          <div className="flex items-center gap-2">
                            <div className="flex gap-1">
                              <div className="w-2 h-2 bg-subtle rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                              <div className="w-2 h-2 bg-subtle rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                              <div className="w-2 h-2 bg-subtle rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                            </div>
                            <span className="text-xs text-subtle font-mono">THINKING...</span>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* AI Response Being Typed */}
                    {aiResponseText && !showMessages.message2 && (
                      <div className="flex justify-start animate-slide-in-left">
                        <div className="bg-paper text-ink px-4 py-2.5 text-sm max-w-xs border-2 border-ink">
                          {aiResponseText}
                          <span className="animate-blink">|</span>
                        </div>
                      </div>
                    )}

                    {/* AI Response Complete */}
                    {showMessages.message2 && (
                      <div className="flex justify-start animate-slide-in-left">
                        <div className="bg-paper text-ink px-4 py-2.5 text-sm max-w-xs border-2 border-ink">
                          Based on <span className="font-mono font-semibold text-accent">[Page 3]</span>, the main hypothesis is that machine learning algorithms can significantly improve prediction accuracy...
                        </div>
                      </div>
                    )}

                    {/* Second User Message */}
                    {showMessages.message3 && (
                      <div className="flex justify-end animate-slide-in-right">
                        <div className="bg-ink text-paper px-4 py-2.5 text-sm max-w-xs border-2 border-ink">
                          {DEMO_MESSAGES[2]}
                        </div>
                      </div>
                    )}
                  </div>
                  <div className="flex gap-2 pt-4 border-t-2 border-ink flex-shrink-0">
                    <input
                      type="text"
                      placeholder="Ask about the document..."
                      value={isTyping ? typingText : ""}
                      className="flex-1 px-4 py-2.5 border-2 border-ink text-sm focus:outline-none focus:border-accent font-serif"
                      readOnly
                    />
                    <button className="bg-ink text-paper px-5 py-2.5 border-2 border-ink hover:bg-accent hover:border-accent transition-colors font-mono text-xs">
                      SEND →
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
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
            <a href="/" className="font-mono text-xl font-bold tracking-tight">
              STUDYFETCH<span className="text-accent">.</span>AI
            </a>
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

      {/* Hero Section */}
      <section className="min-h-screen bg-paper flex flex-col md:flex-row border-b-3 border-ink">
        {/* Left: Giant typographic statement */}
        <div className="w-full md:w-3/5 p-8 md:p-16 flex flex-col justify-center border-b-3 md:border-b-0 md:border-r border-ink">
          <div className="mb-8">
            <span className="font-mono text-xs uppercase tracking-widest text-accent">
              [001] AI-Powered Learning Platform
            </span>
          </div>
          <h1 className="font-mono text-5xl md:text-7xl lg:text-8xl font-bold leading-[0.85] mb-8">
            LEARN<br/>
            <span className="text-accent">SMARTER</span><br/>
            WITH DOCUMENTS
          </h1>
          <p className="font-serif text-xl md:text-2xl leading-relaxed max-w-xl text-subtle mb-12">
            Transform your PDFs into an intelligent study companion. Chat with documents,
            get instant explanations, and accelerate your learning.
          </p>
          <div className="flex flex-col sm:flex-row gap-4">
            <a href="/register" className="font-mono text-sm uppercase bg-ink text-paper px-8 py-4 border-2 border-ink hover:bg-paper hover:text-ink transition-colors text-center">
              Start Learning Free →
            </a>
            <a href="#demo" className="font-mono text-sm uppercase bg-transparent text-ink px-8 py-4 border-2 border-ink hover:bg-accent hover:border-accent hover:text-paper transition-colors text-center">
              ▶ Watch Demo
            </a>
          </div>
        </div>

        {/* Right: Interactive preview / decorative element */}
        <div className="w-full md:w-2/5 bg-ink text-paper p-8 md:p-16 flex items-center justify-center relative overflow-hidden">
          {/* Grid pattern overlay */}
          <div className="absolute inset-0 opacity-10" style={{
            backgroundImage: `linear-gradient(to right, #f5f2eb 1px, transparent 1px), linear-gradient(to bottom, #f5f2eb 1px, transparent 1px)`,
            backgroundSize: '24px 24px'
          }}></div>
          {/* Content: Document preview mockup or typographic element */}
          <div className="relative z-10 text-center">
            <div className="font-mono text-6xl md:text-8xl font-bold opacity-20">PDF</div>
            <div className="font-mono text-xs mt-4 tracking-widest">→ INTELLIGENCE</div>
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
                  <p className="font-serif text-subtle">Advanced vector search finds relevant content even when you don't remember exact keywords.</p>
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

      {/* Demo Section */}
      <AnimatedDemo />

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
