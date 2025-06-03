// app/page.tsx
"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

export default function Home() {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // check if user is authenticated
    const checkAuth = async () => {
      try {
        const response = await fetch('/api/auth/verify-session');

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
      <div className="h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-soid border-blue-500 border-r-transparent"></div>
          <p className="mt-4 text-gray-600">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="scroll-smooth">
      {/* Navigation */}
      <nav className="fixed w-full z-50 bg-white/90 backdrop-blur-md border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <h1 className="text-2xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
                  StudyFetch
                </h1>
              </div>
            </div>
            <div className="hidden md:block">
              <div className="ml-10 flex items-baseline space-x-4">
                <a href="#features" className="text-gray-700 hover:text-purple-600 px-3 py-2 rounded-md text-sm font-medium transition-colors">Features</a>
                <a href="#how-it-works" className="text-gray-700 hover:text-purple-600 px-3 py-2 rounded-md text-sm font-medium transition-colors">How it Works</a>
                <a href="#demo" className="text-gray-700 hover:text-purple-600 px-3 py-2 rounded-md text-sm font-medium transition-colors">Demo</a>
                <a href="/login" className="text-gray-700 hover:text-purple-600 px-3 py-2 rounded-md text-sm font-medium transition-colors">Login</a>
                <a href="/register" className="bg-gradient-to-r from-purple-600 to-blue-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:from-purple-700 hover:to-blue-700 transition-all transform hover:scale-105">
                  Get Started
                </a>
              </div>
            </div>
            <div className="md:hidden">
              <button className="text-gray-700 hover:text-purple-600">
                <i className="fas fa-bars text-xl"></i>
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative min-h-screen flex items-center bg-gradient-to-r from-purple-600 to-blue-600 overflow-hidden">
        {/* Animated Background Elements */}
        <div className="absolute top-20 left-10 w-32 h-32 bg-gradient-to-r from-purple-500 to-blue-500 rounded-full opacity-20 animate-float"></div>
        <div className="absolute bottom-20 right-10 w-24 h-24 bg-gradient-to-r from-purple-500 to-blue-500 rounded-full opacity-15 animate-float" style={{ animationDelay: '-2s' }}></div>
        <div className="absolute top-1/2 right-1/4 w-40 h-40 bg-gradient-to-r from-purple-500 to-blue-500 rounded-full opacity-10 animate-float" style={{ animationDelay: '-4s' }}></div>
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <div className="max-w-4xl mx-auto">
            <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 leading-tight">
              Learn Smarter with <br />
              <span className="bg-gradient-to-r from-yellow-300 to-pink-300 bg-clip-text text-transparent">
                AI-Powered
              </span>
              Tutoring
            </h1>
            <p className="text-xl md:text-2xl text-white/90 mb-8 leading-relaxed">
              Upload your PDFs and chat with an intelligent AI that understands your documents. 
              Get instant answers, explanations, and study help tailored to your materials.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <a href="/register" className="bg-white text-purple-600 px-8 py-4 rounded-xl text-lg font-semibold hover:bg-gray-100 transition-all transform hover:scale-105 shadow-2xl">
                Start Learning Free
                <i className="fas fa-arrow-right ml-2"></i>
              </a>
              <a href="#demo" className="border-2 border-white text-white px-8 py-4 rounded-xl text-lg font-semibold hover:bg-white hover:text-purple-600 transition-all">
                Watch Demo
                <i className="fas fa-play ml-2"></i>
              </a>
            </div>
            <div className="mt-12 text-white/80">
              <p className="text-sm">✨ No credit card required • 🔒 Your data stays secure • ⚡ Instant setup</p>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
              Powerful Features for 
              <span className="bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">Smart Learning</span>
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Everything you need to transform your study materials into an interactive learning experience
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            <div className="backdrop-blur-md bg-white/10 border border-gray-200 rounded-2xl p-8 text-center hover:transform hover:-translate-y-1 transition-all duration-300">
              <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-blue-500 rounded-2xl flex items-center justify-center mx-auto mb-6">
                <i className="fas fa-file-pdf text-white text-2xl"></i>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-4">PDF Upload & Processing</h3>
              <p className="text-gray-600">
                Simply upload your PDF documents and our AI instantly processes them, making every page searchable and interactive.
              </p>
            </div>
            
            <div className="backdrop-blur-md bg-white/10 border border-gray-200 rounded-2xl p-8 text-center hover:transform hover:-translate-y-1 transition-all duration-300">
              <div className="w-16 h-16 bg-gradient-to-r from-green-500 to-teal-500 rounded-2xl flex items-center justify-center mx-auto mb-6">
                <i className="fas fa-comments text-white text-2xl"></i>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-4">Intelligent Chat Interface</h3>
              <p className="text-gray-600">
                Ask questions about your documents in natural language. Get detailed explanations with exact page references.
              </p>
            </div>
            
            <div className="backdrop-blur-md bg-white/10 border border-gray-200 rounded-2xl p-8 text-center hover:transform hover:-translate-y-1 transition-all duration-300">
              <div className="w-16 h-16 bg-gradient-to-r from-orange-500 to-red-500 rounded-2xl flex items-center justify-center mx-auto mb-6">
                <i className="fas fa-search text-white text-2xl"></i>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-4">Smart Document Search</h3>
              <p className="text-gray-600">
                Advanced vector search finds relevant content across your documents, even when you don&apos;t remember exact keywords.
              </p>
            </div>
            
            <div className="backdrop-blur-md bg-white/10 border border-gray-200 rounded-2xl p-8 text-center hover:transform hover:-translate-y-1 transition-all duration-300">
              <div className="w-16 h-16 bg-gradient-to-r from-pink-500 to-purple-500 rounded-2xl flex items-center justify-center mx-auto mb-6">
                <i className="fas fa-bookmark text-white text-2xl"></i>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-4">Persistent Conversations</h3>
              <p className="text-gray-600">
                Your chat history is automatically saved. Pick up where you left off and build on previous discussions.
              </p>
            </div>
            
            <div className="backdrop-blur-md bg-white/10 border border-gray-200 rounded-2xl p-8 text-center hover:transform hover:-translate-y-1 transition-all duration-300">
              <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-indigo-500 rounded-2xl flex items-center justify-center mx-auto mb-6">
                <i className="fas fa-layer-group text-white text-2xl"></i>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-4">Multi-Document Support</h3>
              <p className="text-gray-600">
                Manage multiple documents with separate conversation histories for each. Perfect for different subjects or projects.
              </p>
            </div>
            
            <div className="backdrop-blur-md bg-white/10 border border-gray-200 rounded-2xl p-8 text-center hover:transform hover:-translate-y-1 transition-all duration-300">
              <div className="w-16 h-16 bg-gradient-to-r from-yellow-500 to-orange-500 rounded-2xl flex items-center justify-center mx-auto mb-6">
                <i className="fas fa-shield-alt text-white text-2xl"></i>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-4">Secure & Private</h3>
              <p className="text-gray-600">
                Your documents and conversations are encrypted and stored securely. Only you have access to your data.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* How it Works Section */}
      <section id="how-it-works" className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
              How It Works
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Get started in three simple steps and revolutionize your learning experience
            </p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="relative">
                <div className="w-24 h-24 bg-gradient-to-r from-purple-500 to-blue-500 rounded-full flex items-center justify-center mx-auto mb-6">
                  <span className="text-white text-2xl font-bold">1</span>
                </div>
                <div className="hidden md:block absolute top-12 left-full w-full h-0.5 bg-gradient-to-r from-purple-300 to-blue-300"></div>
              </div>
              <h3 className="text-2xl font-semibold text-gray-900 mb-4">Upload Your PDF</h3>
              <p className="text-gray-600">
                Simply drag and drop your study materials, textbooks, or research papers. Our AI instantly processes them.
              </p>
            </div>
            
            <div className="text-center">
              <div className="relative">
                <div className="w-24 h-24 bg-gradient-to-r from-green-500 to-teal-500 rounded-full flex items-center justify-center mx-auto mb-6">
                  <span className="text-white text-2xl font-bold">2</span>
                </div>
                <div className="hidden md:block absolute top-12 left-full w-full h-0.5 bg-gradient-to-r from-green-300 to-teal-300"></div>
              </div>
              <h3 className="text-2xl font-semibold text-gray-900 mb-4">Ask Questions</h3>
              <p className="text-gray-600">
                Start chatting with your document. Ask for explanations, summaries, or specific information in natural language.
              </p>
            </div>
            
            <div className="text-center">
              <div className="w-24 h-24 bg-gradient-to-r from-orange-500 to-red-500 rounded-full flex items-center justify-center mx-auto mb-6">
                <span className="text-white text-2xl font-bold">3</span>
              </div>
              <h3 className="text-2xl font-semibold text-gray-900 mb-4">Get Smart Answers</h3>
              <p className="text-gray-600">
                Receive detailed, contextual answers with exact page references. Learn faster and understand better.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Demo Section */}
      <section id="demo" className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
              See It In Action
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Watch how StudyFetch transforms your learning experience
            </p>
          </div>
          
          <div className="max-w-4xl mx-auto">
            <div className="demo-screen rounded-2xl p-8">
              <div className="flex gap-6">
                {/* PDF Viewer Mockup */}
                <div className="w-1/2 bg-white rounded-lg p-4 shadow-inner">
                  <div className="flex items-center justify-between mb-4 pb-2 border-b">
                    <span className="text-sm text-gray-600">📄 Research Paper.pdf</span>
                    <span className="text-sm text-gray-500">Page 1 of 15</span>
                  </div>
                  <div className="space-y-3">
                    <div className="h-3 bg-gray-200 rounded w-full"></div>
                    <div className="h-3 bg-gray-200 rounded w-5/6"></div>
                    <div className="h-3 bg-purple-200 rounded w-4/6"></div>
                    <div className="h-3 bg-gray-200 rounded w-full"></div>
                    <div className="h-3 bg-gray-200 rounded w-3/4"></div>
                    <div className="h-8 bg-gray-100 rounded"></div>
                    <div className="h-3 bg-gray-200 rounded w-full"></div>
                    <div className="h-3 bg-purple-200 rounded w-5/6"></div>
                    <div className="h-3 bg-gray-200 rounded w-2/3"></div>
                  </div>
                </div>
                
                {/* Chat Interface Mockup */}
                <div className="w-1/2 bg-white rounded-lg p-4 shadow-inner">
                  <div className="border-b pb-2 mb-4">
                    <span className="text-sm font-medium text-gray-700">💬 AI Tutor Chat</span>
                  </div>
                  <div className="space-y-4 h-48 overflow-hidden">
                    <div className="flex justify-end">
                      <div className="chat-bubble-user text-white px-4 py-2 rounded-lg text-sm max-w-xs">
                        Can you explain the main hypothesis in this paper?
                      </div>
                    </div>
                    <div className="flex justify-start">
                      <div className="chat-bubble-ai text-gray-700 px-4 py-2 rounded-lg text-sm max-w-xs">
                        Based on [Page 3], the main hypothesis is that machine learning algorithms can significantly improve prediction accuracy...
                      </div>
                    </div>
                    <div className="flex justify-end">
                      <div className="chat-bubble-user text-white px-4 py-2 rounded-lg text-sm max-w-xs">
                        What evidence supports this?
                      </div>
                    </div>
                  </div>
                  <div className="mt-4 flex gap-2">
                    <input type="text" placeholder="Ask about the document..." className="flex-1 px-3 py-2 border rounded-lg text-sm" readOnly />
                    <button className="bg-purple-500 text-white px-4 py-2 rounded-lg">
                      <i className="fas fa-paper-plane"></i>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-r from-purple-600 to-blue-600">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
            Ready to Transform Your Learning?
          </h2>
          <p className="text-xl text-white/90 mb-8 max-w-3xl mx-auto">
            Join thousands of students who are already learning smarter with AI-powered tutoring
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <a href="/register" className="bg-white text-purple-600 px-8 py-4 rounded-xl text-lg font-semibold hover:bg-gray-100 transition-all transform hover:scale-105 shadow-2xl">
              Get Started Free
              <i className="fas fa-rocket ml-2"></i>
            </a>
            <a href="/login" className="border-2 border-white text-white px-8 py-4 rounded-xl text-lg font-semibold hover:bg-white hover:text-purple-600 transition-all">
              Sign In
              <i className="fas fa-sign-in-alt ml-2"></i>
            </a>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid md:grid-cols-4 gap-8">
            <div className="col-span-2">
              <h3 className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent mb-4">
                StudyFetch
              </h3>
              <p className="text-gray-300 mb-4">
                Revolutionizing education with AI-powered document interaction and intelligent tutoring.
              </p>
              <div className="flex space-x-4">
                <a href="#" className="text-gray-400 hover:text-white transition-colors">
                  <i className="fab fa-twitter text-xl"></i>
                </a>
                <a href="#" className="text-gray-400 hover:text-white transition-colors">
                  <i className="fab fa-github text-xl"></i>
                </a>
                <a href="#" className="text-gray-400 hover:text-white transition-colors">
                  <i className="fab fa-linkedin text-xl"></i>
                </a>
              </div>
            </div>
            <div>
              <h4 className="text-lg font-semibold mb-4">Product</h4>
              <ul className="space-y-2 text-gray-300">
                <li><a href="#features" className="hover:text-white transition-colors">Features</a></li>
                <li><a href="#demo" className="hover:text-white transition-colors">Demo</a></li>
                <li><a href="/register" className="hover:text-white transition-colors">Get Started</a></li>
              </ul>
            </div>
            <div>
              <h4 className="text-lg font-semibold mb-4">Support</h4>
              <ul className="space-y-2 text-gray-300">
                <li><a href="#" className="hover:text-white transition-colors">Help Center</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Privacy Policy</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Terms of Service</a></li>
              </ul>
            </div>
          </div>
          <div className="border-t border-gray-800 mt-8 pt-8 text-center text-gray-400">
            <p>&copy; 2024 StudyFetch AI Tutor. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
