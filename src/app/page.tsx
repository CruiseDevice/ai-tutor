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
    <section id="demo" className="py-24 bg-gradient-to-b from-gray-50 to-white relative overflow-hidden">
      <div className="absolute top-0 right-0 w-1/2 h-full bg-gradient-to-l from-purple-50/50 to-transparent"></div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <div className="text-center mb-20">
          <div className="inline-block px-4 py-2 rounded-full bg-pink-100 text-pink-700 text-sm font-semibold mb-6">
            INTERACTIVE DEMO
          </div>
          <h2 className="text-5xl md:text-6xl font-extrabold text-gray-900 mb-6">
            See It In Action
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
            Watch how StudyFetch transforms your learning experience
          </p>
        </div>

        <div className="max-w-5xl mx-auto">
          <div className="demo-screen rounded-3xl p-1 shadow-2xl">
            <div className="bg-white rounded-3xl p-8">
              <div className="flex flex-col lg:flex-row gap-6">
                {/* PDF Viewer Mockup */}
                <div ref={documentRef} className="lg:w-1/2 bg-white rounded-2xl p-6 shadow-inner border border-gray-200 overflow-y-auto max-h-[600px] relative" style={{
                  backgroundImage: 'linear-gradient(to bottom, rgba(0,0,0,0.02) 0%, transparent 100%)'
                }}>
                  <div className="flex items-center justify-between mb-4 pb-3 border-b border-gray-300 sticky top-0 bg-white z-10">
                    <div className="flex items-center gap-2">
                      <div className="w-8 h-8 bg-red-500 rounded flex items-center justify-center">
                        <i className="fas fa-file-pdf text-white text-xs"></i>
                      </div>
                      <span className="text-sm font-semibold text-gray-700">Research Paper.pdf</span>
                    </div>
                    <span className="text-xs text-gray-500 bg-gray-200 px-2 py-1 rounded transition-all duration-500">
                      Page {currentPage} of 15
                    </span>
                  </div>
                  <div className="prose prose-sm max-w-none text-gray-800 leading-relaxed">
                    {/* Title */}
                    <h1 className={`text-xl font-bold mb-2 text-gray-900 transition-all duration-500 ${highlightedLine === 0 ? 'bg-purple-100 px-2 py-1 rounded' : ''}`}>
                      Machine Learning Algorithms for Predictive Analytics in Healthcare
                    </h1>

                    {/* Authors */}
                    <div className={`text-xs text-gray-600 mb-4 italic transition-all duration-500 ${highlightedLine === 1 ? 'bg-purple-100 px-2 py-1 rounded' : ''}`}>
                      John A. Smith, Sarah B. Johnson, Michael C. Williams
                    </div>

                    {/* Abstract */}
                    <div ref={abstractRef} className={`mb-4 transition-all duration-500 ${highlightedLine === 2 ? 'bg-purple-200 px-3 py-2 rounded border-l-4 border-purple-500 shadow-md ring-2 ring-purple-400' : ''}`}>
                      <h2 className="text-sm font-semibold text-gray-900 mb-2">Abstract</h2>
                      <p className={`text-xs leading-relaxed mb-2 transition-all duration-500 ${highlightedLine === 2 ? 'font-medium' : ''}`}>
                        This paper presents a comprehensive analysis of machine learning algorithms and their application in healthcare predictive analytics. We evaluate the performance of various models including neural networks, random forests, and support vector machines.
                      </p>
                      <p className={`text-xs leading-relaxed transition-all duration-500 ${highlightedLine === 3 ? 'bg-purple-100 px-2 py-1 rounded' : ''}`}>
                        Our results demonstrate that <span className={`transition-all duration-500 ${highlightedLine === 2 ? 'bg-yellow-300 font-semibold px-1.5 py-0.5 rounded shadow-sm animate-pulse' : ''}`}>machine learning algorithms can significantly improve prediction accuracy</span> compared to traditional statistical methods, with an average improvement of 23.4% across all tested datasets.
                      </p>
                    </div>

                    {/* Introduction */}
                    <div className={`mb-4 transition-all duration-500 ${highlightedLine === 4 ? 'bg-purple-100 px-3 py-2 rounded' : ''}`}>
                      <h2 className="text-sm font-semibold text-gray-900 mb-2">1. Introduction</h2>
                      <p className="text-xs leading-relaxed mb-2">
                        The integration of artificial intelligence and machine learning in healthcare has revolutionized the way we approach medical diagnosis and treatment planning. Recent advances in computational power and data availability have enabled researchers to develop sophisticated predictive models.
                      </p>
                      <p className={`text-xs leading-relaxed transition-all duration-500 ${highlightedLine === 5 ? 'bg-purple-100 px-2 py-1 rounded' : ''}`}>
                        The primary objective of this research is to investigate the effectiveness of different machine learning approaches in predicting patient outcomes and identifying high-risk cases early in the treatment process.
                      </p>
                    </div>

                    {/* Methodology Section */}
                    <div className={`mb-4 transition-all duration-500 ${highlightedLine === 6 ? 'bg-purple-100 px-3 py-2 rounded' : ''}`}>
                      <h2 className="text-sm font-semibold text-gray-900 mb-2">2. Methodology</h2>
                      <p className="text-xs leading-relaxed mb-2">
                        We conducted a systematic review of existing literature and performed empirical analysis on three distinct healthcare datasets. Each dataset contained anonymized patient records spanning a period of five years.
                      </p>
                      <p className={`text-xs leading-relaxed transition-all duration-500 ${highlightedLine === 7 ? 'bg-purple-200 px-2 py-1 rounded border-l-4 border-purple-500' : ''}`}>
                        The experimental setup involved training multiple models using cross-validation techniques to ensure robust performance metrics. We compared accuracy, precision, recall, and F1-scores across all models.
                      </p>
                    </div>

                    {/* Results Preview */}
                    <div className={`mb-2 transition-all duration-500 ${highlightedLine === 8 ? 'bg-purple-100 px-3 py-2 rounded' : ''}`}>
                      <h2 className="text-sm font-semibold text-gray-900 mb-2">3. Results</h2>
                      <p className="text-xs leading-relaxed">
                        Our analysis reveals significant improvements in prediction accuracy when using ensemble methods. The random forest classifier achieved the highest performance with 94.2% accuracy...
                      </p>
                    </div>
                  </div>
                </div>

                {/* Chat Interface Mockup */}
                <div className="lg:w-1/2 bg-white rounded-2xl p-6 shadow-inner border border-gray-200 flex flex-col max-h-[600px]">
                  <div className="flex items-center gap-2 border-b border-gray-200 pb-3 mb-4 flex-shrink-0">
                    <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-blue-500 rounded-full flex items-center justify-center">
                      <i className="fas fa-robot text-white text-xs"></i>
                    </div>
                    <span className="text-sm font-semibold text-gray-700">AI Tutor Chat</span>
                  </div>
                  <div className="space-y-4 flex-1 overflow-y-auto mb-4 min-h-0">
                    {/* First User Message */}
                    {showMessages.message1 && (
                      <div className="flex justify-end animate-slide-in-right">
                        <div className="chat-bubble-user text-white px-4 py-2.5 rounded-2xl text-sm max-w-xs shadow-md">
                          {DEMO_MESSAGES[0]}
                        </div>
                      </div>
                    )}

                    {/* AI Thinking Indicator */}
                    {isAIThinking && (
                      <div className="flex justify-start animate-slide-in-left">
                        <div className="chat-bubble-ai text-gray-700 px-4 py-2.5 rounded-2xl text-sm max-w-xs shadow-md">
                          <div className="flex items-center gap-2">
                            <div className="flex gap-1">
                              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                            </div>
                            <span className="text-xs text-gray-500">AI is thinking...</span>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* AI Response Being Typed */}
                    {aiResponseText && !showMessages.message2 && (
                      <div className="flex justify-start animate-slide-in-left">
                        <div className="chat-bubble-ai text-gray-700 px-4 py-2.5 rounded-2xl text-sm max-w-xs shadow-md">
                          {aiResponseText}
                          <span className="animate-blink">|</span>
                        </div>
                      </div>
                    )}

                    {/* AI Response Complete */}
                    {showMessages.message2 && (
                      <div className="flex justify-start animate-slide-in-left">
                        <div className="chat-bubble-ai text-gray-700 px-4 py-2.5 rounded-2xl text-sm max-w-xs shadow-md">
                          Based on <span className="font-semibold text-purple-600 animate-highlight">[Page 3]</span>, the main hypothesis is that machine learning algorithms can significantly improve prediction accuracy...
                        </div>
                      </div>
                    )}

                    {/* Second User Message */}
                    {showMessages.message3 && (
                      <div className="flex justify-end animate-slide-in-right">
                        <div className="chat-bubble-user text-white px-4 py-2.5 rounded-2xl text-sm max-w-xs shadow-md">
                          {DEMO_MESSAGES[2]}
                        </div>
                      </div>
                    )}
                  </div>
                  <div className="flex gap-2 pt-4 border-t border-gray-200 flex-shrink-0">
                    <input
                      type="text"
                      placeholder="Ask about the document..."
                      value={isTyping ? typingText : ""}
                      className="flex-1 px-4 py-2.5 border border-gray-300 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
                      readOnly
                    />
                    <button className="bg-gradient-to-r from-purple-500 to-blue-500 text-white px-5 py-2.5 rounded-xl hover:shadow-lg transition-all">
                      <i className="fas fa-paper-plane"></i>
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
      <div className="h-screen flex items-center justify-center bg-gradient-to-br from-purple-50 via-blue-50 to-pink-50">
        <div className="text-center">
          <div className="relative inline-block">
            <div className="absolute inset-0 rounded-full bg-gradient-to-r from-purple-600 to-blue-600 opacity-20 animate-pulse-glow"></div>
            <div className="relative inline-block h-12 w-12 animate-spin rounded-full border-4 border-purple-200 border-t-purple-600 border-r-blue-600"></div>
          </div>
          <p className="mt-6 text-gray-700 font-medium">Loading your learning experience...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="scroll-smooth bg-white">
      {/* Navigation */}
      <nav className="fixed w-full z-50 bg-white/95 backdrop-blur-md border-b border-gray-200/50 shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-20">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="flex items-center space-x-2">
                  <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-600 via-blue-600 to-pink-600 flex items-center justify-center shadow-lg">
                    <span className="text-white font-bold text-xl">SF</span>
                  </div>
                  <h1 className="text-2xl font-bold gradient-text">
                    StudyFetch
                  </h1>
                </div>
              </div>
            </div>
            <div className="hidden md:block">
              <div className="ml-10 flex items-baseline space-x-1">
                <a href="#features" className="text-gray-900 hover:text-purple-600 px-4 py-2 rounded-lg text-sm font-semibold transition-all hover:bg-purple-50">Features</a>
                <a href="#how-it-works" className="text-gray-900 hover:text-purple-600 px-4 py-2 rounded-lg text-sm font-semibold transition-all hover:bg-purple-50">How it Works</a>
                <a href="#demo" className="text-gray-900 hover:text-purple-600 px-4 py-2 rounded-lg text-sm font-semibold transition-all hover:bg-purple-50">Demo</a>
                <a href="/login" className="text-gray-900 hover:text-purple-600 px-4 py-2 rounded-lg text-sm font-semibold transition-all hover:bg-purple-50">Login</a>
                <a href="/register" className="ml-2 bg-gradient-to-r from-purple-600 via-blue-600 to-pink-600 text-white px-6 py-2 rounded-xl text-sm font-semibold hover:shadow-xl transition-all transform hover:scale-105 animate-gradient">
                  Get Started
                </a>
              </div>
            </div>
            <div className="md:hidden">
              <button className="text-gray-900 hover:text-purple-600 p-2 rounded-lg hover:bg-purple-50 transition-all">
                <i className="fas fa-bars text-xl"></i>
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative min-h-screen flex items-center bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 overflow-hidden pt-20">
        {/* Animated Background Elements */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute top-20 left-10 w-72 h-72 bg-gradient-to-r from-purple-500/30 to-pink-500/30 rounded-full blur-3xl animate-float"></div>
          <div className="absolute bottom-20 right-10 w-96 h-96 bg-gradient-to-r from-blue-500/30 to-cyan-500/30 rounded-full blur-3xl animate-float" style={{ animationDelay: '-2s' }}></div>
          <div className="absolute top-1/2 right-1/4 w-80 h-80 bg-gradient-to-r from-indigo-500/20 to-purple-500/20 rounded-full blur-3xl animate-float" style={{ animationDelay: '-4s' }}></div>
          <div className="absolute top-1/3 left-1/3 w-64 h-64 bg-gradient-to-r from-pink-500/20 to-rose-500/20 rounded-full blur-3xl animate-float" style={{ animationDelay: '-1s' }}></div>
        </div>

        {/* Grid Pattern Overlay */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]"></div>

        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center z-10">
          <div className="max-w-5xl mx-auto animate-slide-up">
            {/* Badge */}
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/10 backdrop-blur-md border border-white/20 mb-8">
              <span className="text-yellow-300 text-xl">✨</span>
              <span className="text-white/90 text-sm font-medium">AI-Powered Learning Platform</span>
            </div>

            <h1 className="text-6xl md:text-8xl font-extrabold text-white mb-8 leading-tight tracking-tight">
              Learn Smarter with <br />
              <span className="relative inline-block">
                <span className="relative z-10 bg-gradient-to-r from-yellow-300 via-pink-300 to-purple-300 bg-clip-text text-transparent animate-gradient">
                  AI-Powered
                </span>
                <span className="absolute inset-0 bg-gradient-to-r from-yellow-300/20 via-pink-300/20 to-purple-300/20 blur-2xl"></span>
              </span>
              <br />Tutoring
            </h1>
            <p className="text-xl md:text-2xl text-white/80 mb-10 leading-relaxed max-w-3xl mx-auto">
              Transform your PDFs into an intelligent study companion. Chat with documents, get instant explanations,
              and accelerate your learning with AI that truly understands your materials.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-12">
              <a href="/register" className="group relative bg-white text-purple-600 px-10 py-5 rounded-2xl text-lg font-bold hover:shadow-2xl transition-all transform hover:scale-105 overflow-hidden">
                <span className="relative z-10 flex items-center">
                  Start Learning Free
                  <i className="fas fa-arrow-right ml-3 group-hover:translate-x-1 transition-transform"></i>
                </span>
                <div className="absolute inset-0 bg-gradient-to-r from-purple-100 to-pink-100 opacity-0 group-hover:opacity-100 transition-opacity"></div>
              </a>
              <a href="#demo" className="group border-2 border-white/30 text-white px-10 py-5 rounded-2xl text-lg font-semibold hover:bg-white/10 hover:border-white/50 transition-all backdrop-blur-sm">
                <span className="flex items-center">
                  <i className="fas fa-play mr-3 group-hover:scale-110 transition-transform"></i>
                  Watch Demo
                </span>
              </a>
            </div>
            <div className="flex flex-wrap justify-center gap-8 text-white/70 text-sm">
              <div className="flex items-center gap-2">
                <span className="text-xl">✨</span>
                <span>No credit card required</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xl">🔒</span>
                <span>Your data stays secure</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xl">⚡</span>
                <span>Instant setup</span>
              </div>
            </div>
          </div>
        </div>

        {/* Scroll Indicator */}
        <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 animate-bounce">
          <div className="w-6 h-10 border-2 border-white/30 rounded-full flex justify-center">
            <div className="w-1 h-3 bg-white/50 rounded-full mt-2"></div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-24 bg-gradient-to-b from-white to-gray-50 relative overflow-hidden">
        {/* Decorative elements */}
        <div className="absolute top-0 left-0 w-96 h-96 bg-purple-200/20 rounded-full blur-3xl"></div>
        <div className="absolute bottom-0 right-0 w-96 h-96 bg-blue-200/20 rounded-full blur-3xl"></div>

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <div className="text-center mb-20">
            <div className="inline-block px-4 py-2 rounded-full bg-purple-100 text-purple-700 text-sm font-semibold mb-6">
              POWERFUL FEATURES
            </div>
            <h2 className="text-5xl md:text-6xl font-extrabold text-gray-900 mb-6">
              Everything You Need for
              <br />
              <span className="gradient-text">Smart Learning</span>
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
              Transform your study materials into an interactive, intelligent learning experience
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div className="feature-card group bg-white rounded-3xl p-8 border border-gray-100 shadow-lg hover:shadow-2xl">
              <div className="relative mb-6">
                <div className="absolute inset-0 bg-gradient-to-r from-purple-500 to-blue-500 rounded-2xl blur-lg opacity-0 group-hover:opacity-30 transition-opacity"></div>
                <div className="relative w-16 h-16 bg-gradient-to-br from-purple-500 to-blue-500 rounded-2xl flex items-center justify-center shadow-lg">
                  <i className="fas fa-file-pdf text-white text-2xl"></i>
                </div>
              </div>
              <h3 className="text-2xl font-bold text-gray-900 mb-4">PDF Upload & Processing</h3>
              <p className="text-gray-600 leading-relaxed">
                Simply upload your PDF documents and our AI instantly processes them, making every page searchable and interactive.
              </p>
            </div>

            <div className="feature-card group bg-white rounded-3xl p-8 border border-gray-100 shadow-lg hover:shadow-2xl">
              <div className="relative mb-6">
                <div className="absolute inset-0 bg-gradient-to-r from-green-500 to-teal-500 rounded-2xl blur-lg opacity-0 group-hover:opacity-30 transition-opacity"></div>
                <div className="relative w-16 h-16 bg-gradient-to-br from-green-500 to-teal-500 rounded-2xl flex items-center justify-center shadow-lg">
                  <i className="fas fa-comments text-white text-2xl"></i>
                </div>
              </div>
              <h3 className="text-2xl font-bold text-gray-900 mb-4">Intelligent Chat Interface</h3>
              <p className="text-gray-600 leading-relaxed">
                Ask questions about your documents in natural language. Get detailed explanations with exact page references.
              </p>
            </div>

            <div className="feature-card group bg-white rounded-3xl p-8 border border-gray-100 shadow-lg hover:shadow-2xl">
              <div className="relative mb-6">
                <div className="absolute inset-0 bg-gradient-to-r from-orange-500 to-red-500 rounded-2xl blur-lg opacity-0 group-hover:opacity-30 transition-opacity"></div>
                <div className="relative w-16 h-16 bg-gradient-to-br from-orange-500 to-red-500 rounded-2xl flex items-center justify-center shadow-lg">
                  <i className="fas fa-search text-white text-2xl"></i>
                </div>
              </div>
              <h3 className="text-2xl font-bold text-gray-900 mb-4">Smart Document Search</h3>
              <p className="text-gray-600 leading-relaxed">
                Advanced vector search finds relevant content across your documents, even when you don&apos;t remember exact keywords.
              </p>
            </div>

            <div className="feature-card group bg-white rounded-3xl p-8 border border-gray-100 shadow-lg hover:shadow-2xl">
              <div className="relative mb-6">
                <div className="absolute inset-0 bg-gradient-to-r from-pink-500 to-purple-500 rounded-2xl blur-lg opacity-0 group-hover:opacity-30 transition-opacity"></div>
                <div className="relative w-16 h-16 bg-gradient-to-br from-pink-500 to-purple-500 rounded-2xl flex items-center justify-center shadow-lg">
                  <i className="fas fa-bookmark text-white text-2xl"></i>
                </div>
              </div>
              <h3 className="text-2xl font-bold text-gray-900 mb-4">Persistent Conversations</h3>
              <p className="text-gray-600 leading-relaxed">
                Your chat history is automatically saved. Pick up where you left off and build on previous discussions.
              </p>
            </div>

            <div className="feature-card group bg-white rounded-3xl p-8 border border-gray-100 shadow-lg hover:shadow-2xl">
              <div className="relative mb-6">
                <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-indigo-500 rounded-2xl blur-lg opacity-0 group-hover:opacity-30 transition-opacity"></div>
                <div className="relative w-16 h-16 bg-gradient-to-br from-blue-500 to-indigo-500 rounded-2xl flex items-center justify-center shadow-lg">
                  <i className="fas fa-layer-group text-white text-2xl"></i>
                </div>
              </div>
              <h3 className="text-2xl font-bold text-gray-900 mb-4">Multi-Document Support</h3>
              <p className="text-gray-600 leading-relaxed">
                Manage multiple documents with separate conversation histories for each. Perfect for different subjects or projects.
              </p>
            </div>

            <div className="feature-card group bg-white rounded-3xl p-8 border border-gray-100 shadow-lg hover:shadow-2xl">
              <div className="relative mb-6">
                <div className="absolute inset-0 bg-gradient-to-r from-yellow-500 to-orange-500 rounded-2xl blur-lg opacity-0 group-hover:opacity-30 transition-opacity"></div>
                <div className="relative w-16 h-16 bg-gradient-to-br from-yellow-500 to-orange-500 rounded-2xl flex items-center justify-center shadow-lg">
                  <i className="fas fa-shield-alt text-white text-2xl"></i>
                </div>
              </div>
              <h3 className="text-2xl font-bold text-gray-900 mb-4">Secure & Private</h3>
              <p className="text-gray-600 leading-relaxed">
                Your documents and conversations are encrypted and stored securely. Only you have access to your data.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* How it Works Section */}
      <section id="how-it-works" className="py-24 bg-white relative overflow-hidden">
        {/* Background Pattern */}
        <div className="absolute inset-0 opacity-5">
          <div className="absolute inset-0" style={{
            backgroundImage: `radial-gradient(circle at 2px 2px, rgb(139, 92, 246) 1px, transparent 0)`,
            backgroundSize: '40px 40px'
          }}></div>
        </div>

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <div className="text-center mb-20">
            <div className="inline-block px-4 py-2 rounded-full bg-blue-100 text-blue-700 text-sm font-semibold mb-6">
              SIMPLE PROCESS
            </div>
            <h2 className="text-5xl md:text-6xl font-extrabold text-gray-900 mb-6">
              How It Works
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
              Get started in three simple steps and revolutionize your learning experience
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-12 relative">
            {/* Connection Line */}
            <div className="hidden md:block absolute top-24 left-1/3 right-1/3 h-1 bg-gradient-to-r from-purple-300 via-blue-300 to-green-300"></div>

            <div className="text-center relative">
              <div className="relative inline-block mb-8">
                <div className="absolute inset-0 bg-gradient-to-r from-purple-500 to-blue-500 rounded-full blur-xl opacity-30 animate-pulse"></div>
                <div className="relative w-28 h-28 bg-gradient-to-br from-purple-500 to-blue-500 rounded-full flex items-center justify-center shadow-2xl border-4 border-white">
                  <span className="text-white text-4xl font-bold">1</span>
                </div>
              </div>
              <div className="bg-white rounded-2xl p-6 shadow-lg border border-gray-100">
                <h3 className="text-2xl font-bold text-gray-900 mb-4">Upload Your PDF</h3>
                <p className="text-gray-600 leading-relaxed">
                  Simply drag and drop your study materials, textbooks, or research papers. Our AI instantly processes them.
                </p>
              </div>
            </div>

            <div className="text-center relative">
              <div className="relative inline-block mb-8">
                <div className="absolute inset-0 bg-gradient-to-r from-green-500 to-teal-500 rounded-full blur-xl opacity-30 animate-pulse" style={{ animationDelay: '0.5s' }}></div>
                <div className="relative w-28 h-28 bg-gradient-to-br from-green-500 to-teal-500 rounded-full flex items-center justify-center shadow-2xl border-4 border-white">
                  <span className="text-white text-4xl font-bold">2</span>
                </div>
              </div>
              <div className="bg-white rounded-2xl p-6 shadow-lg border border-gray-100">
                <h3 className="text-2xl font-bold text-gray-900 mb-4">Ask Questions</h3>
                <p className="text-gray-600 leading-relaxed">
                  Start chatting with your document. Ask for explanations, summaries, or specific information in natural language.
                </p>
              </div>
            </div>

            <div className="text-center relative">
              <div className="relative inline-block mb-8">
                <div className="absolute inset-0 bg-gradient-to-r from-orange-500 to-red-500 rounded-full blur-xl opacity-30 animate-pulse" style={{ animationDelay: '1s' }}></div>
                <div className="relative w-28 h-28 bg-gradient-to-br from-orange-500 to-red-500 rounded-full flex items-center justify-center shadow-2xl border-4 border-white">
                  <span className="text-white text-4xl font-bold">3</span>
                </div>
              </div>
              <div className="bg-white rounded-2xl p-6 shadow-lg border border-gray-100">
                <h3 className="text-2xl font-bold text-gray-900 mb-4">Get Smart Answers</h3>
                <p className="text-gray-600 leading-relaxed">
                  Receive detailed, contextual answers with exact page references. Learn faster and understand better.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Demo Section */}
      <AnimatedDemo />

      {/* CTA Section */}
      <section className="py-24 bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 relative overflow-hidden">
        {/* Animated Background */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute top-0 left-1/4 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-float"></div>
          <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-blue-500/20 rounded-full blur-3xl animate-float" style={{ animationDelay: '-2s' }}></div>
        </div>

        {/* Grid Pattern */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#ffffff08_1px,transparent_1px),linear-gradient(to_bottom,#ffffff08_1px,transparent_1px)] bg-[size:24px_24px]"></div>

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center relative z-10">
          <div className="max-w-4xl mx-auto">
            <div className="inline-block px-4 py-2 rounded-full bg-white/10 backdrop-blur-md border border-white/20 text-white text-sm font-semibold mb-8">
              JOIN THE FUTURE OF LEARNING
            </div>
            <h2 className="text-5xl md:text-7xl font-extrabold text-white mb-8 leading-tight">
              Ready to Transform Your Learning?
            </h2>
            <p className="text-xl md:text-2xl text-white/90 mb-12 max-w-3xl mx-auto leading-relaxed">
              Join thousands of students who are already learning smarter with AI-powered tutoring
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <a href="/register" className="group relative bg-white text-purple-600 px-10 py-5 rounded-2xl text-lg font-bold hover:shadow-2xl transition-all transform hover:scale-105 overflow-hidden">
                <span className="relative z-10 flex items-center">
                  Get Started Free
                  <i className="fas fa-rocket ml-3 group-hover:translate-x-1 group-hover:-translate-y-1 transition-transform"></i>
                </span>
                <div className="absolute inset-0 bg-gradient-to-r from-purple-100 to-pink-100 opacity-0 group-hover:opacity-100 transition-opacity"></div>
              </a>
              <a href="/login" className="group border-2 border-white/30 text-white px-10 py-5 rounded-2xl text-lg font-semibold hover:bg-white/10 hover:border-white/50 transition-all backdrop-blur-sm">
                <span className="flex items-center">
                  Sign In
                  <i className="fas fa-sign-in-alt ml-3 group-hover:translate-x-1 transition-transform"></i>
                </span>
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-16 relative overflow-hidden">
        {/* Background Pattern */}
        <div className="absolute inset-0 opacity-5">
          <div className="absolute inset-0" style={{
            backgroundImage: `radial-gradient(circle at 2px 2px, rgb(255, 255, 255) 1px, transparent 0)`,
            backgroundSize: '40px 40px'
          }}></div>
        </div>

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <div className="grid md:grid-cols-4 gap-12 mb-12">
            <div className="col-span-2">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-purple-600 via-blue-600 to-pink-600 flex items-center justify-center shadow-lg">
                  <span className="text-white font-bold text-xl">SF</span>
                </div>
                <h3 className="text-2xl font-bold gradient-text">
                  StudyFetch
                </h3>
              </div>
              <p className="text-gray-300 mb-6 leading-relaxed max-w-md">
                Revolutionizing education with AI-powered document interaction and intelligent tutoring.
                Learn smarter, faster, and more effectively.
              </p>
              <div className="flex space-x-4">
                <a href="#" className="w-10 h-10 rounded-lg bg-gray-800 hover:bg-gradient-to-br hover:from-purple-600 hover:to-blue-600 flex items-center justify-center text-gray-400 hover:text-white transition-all transform hover:scale-110">
                  <i className="fab fa-twitter"></i>
                </a>
                <a href="#" className="w-10 h-10 rounded-lg bg-gray-800 hover:bg-gradient-to-br hover:from-purple-600 hover:to-blue-600 flex items-center justify-center text-gray-400 hover:text-white transition-all transform hover:scale-110">
                  <i className="fab fa-github"></i>
                </a>
                <a href="#" className="w-10 h-10 rounded-lg bg-gray-800 hover:bg-gradient-to-br hover:from-purple-600 hover:to-blue-600 flex items-center justify-center text-gray-400 hover:text-white transition-all transform hover:scale-110">
                  <i className="fab fa-linkedin"></i>
                </a>
              </div>
            </div>
            <div>
              <h4 className="text-lg font-bold mb-6 text-white">Product</h4>
              <ul className="space-y-3 text-gray-300">
                <li><a href="#features" className="hover:text-white hover:translate-x-1 inline-block transition-all">Features</a></li>
                <li><a href="#demo" className="hover:text-white hover:translate-x-1 inline-block transition-all">Demo</a></li>
                <li><a href="/register" className="hover:text-white hover:translate-x-1 inline-block transition-all">Get Started</a></li>
              </ul>
            </div>
            <div>
              <h4 className="text-lg font-bold mb-6 text-white">Support</h4>
              <ul className="space-y-3 text-gray-300">
                <li><a href="#" className="hover:text-white hover:translate-x-1 inline-block transition-all">Help Center</a></li>
                <li><a href="#" className="hover:text-white hover:translate-x-1 inline-block transition-all">Privacy Policy</a></li>
                <li><a href="#" className="hover:text-white hover:translate-x-1 inline-block transition-all">Terms of Service</a></li>
              </ul>
            </div>
          </div>
          <div className="border-t border-gray-800 pt-8 flex flex-col md:flex-row justify-between items-center">
            <p className="text-gray-400 text-sm">&copy; 2024 StudyFetch AI Tutor. All rights reserved.</p>
            <div className="flex gap-6 mt-4 md:mt-0">
              <a href="#" className="text-gray-400 hover:text-white text-sm transition-colors">Privacy</a>
              <a href="#" className="text-gray-400 hover:text-white text-sm transition-colors">Terms</a>
              <a href="#" className="text-gray-400 hover:text-white text-sm transition-colors">Cookies</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
