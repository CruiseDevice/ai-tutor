import { Bot, User, FileText, ArrowRight, Copy, Check } from "lucide-react";
import React, { useMemo, useCallback, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import SyntaxHighlighter from 'react-syntax-highlighter/dist/esm/prism';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { useChatStore } from "@/stores/chatStore";
import type { AnnotationReference } from "@/types";

// =====================================================
// HELPER FUNCTIONS
// =====================================================

// Helper function to convert LaTeX delimiters to markdown math format
const preprocessMathContent = (content: string): string => {
  // Split content into parts, separating code blocks and inline code
  const parts: Array<{ type: 'code' | 'text', content: string }> = [];

  // Match code blocks (```...```) and inline code (`...`)
  const codeBlockRegex = /(```[\s\S]*?```|`[^`\n]+?`)/g;
  let lastIndex = 0;
  let match;

  while ((match = codeBlockRegex.exec(content)) !== null) {
    // Add text before code block
    if (match.index > lastIndex) {
      parts.push({ type: 'text', content: content.slice(lastIndex, match.index) });
    }
    // Add code block
    parts.push({ type: 'code', content: match[0] });
    lastIndex = match.index + match[0].length;
  }

  // Add remaining text
  if (lastIndex < content.length) {
    parts.push({ type: 'text', content: content.slice(lastIndex) });
  }

  // Process only text parts, leave code blocks unchanged
  return parts.map(part => {
    if (part.type === 'code') {
      return part.content;
    }

    let processed = part.content;

    // Convert \[ ... \] to $$ ... $$ for display math
    // Use a function replacer to avoid `$1` substitution quirks
    processed = processed.replace(/\\\[([\s\S]*?)\\\]/g, (_match, inner) => `$$${inner}$$`);

    // Convert \( ... \) to $ ... $ for inline math
    processed = processed.replace(/\\\((.*?)\\\)/g, (_match, inner) => `$${inner}$`);

    return processed;
  }).join('');
};

// Code block component with syntax highlighting and copy button
interface CodeBlockProps {
  inline?: boolean;
  className?: string;
  children?: React.ReactNode;
  [key: string]: unknown;
}

const CodeBlock = ({ inline, className, children }: CodeBlockProps) => {
  const [copied, setCopied] = useState(false);
  const match = /language-(\w+)/.exec(className || '');
  const language = match ? match[1] : '';

  // Extract code string from children - ensure it's always a plain string
  // ReactMarkdown passes children as an array of strings for code blocks
  // Use useMemo to ensure we don't recreate the string unnecessarily
  const codeString = useMemo(() => {
    const extracted = React.Children.toArray(children)
      .map((child) => {
        // For code blocks, children should be strings
        if (typeof child === 'string') {
          return child;
        }
        // If it's a React element, try to extract text content
        if (React.isValidElement(child)) {
          const element = child as React.ReactElement<{ children?: React.ReactNode }>;
          if (element.props?.children) {
            return React.Children.toArray(element.props.children)
              .map(c => typeof c === 'string' ? c : String(c))
              .join('');
          }
        }
        // Fallback: convert to string
        return String(child);
      })
      .join('')
      .replace(/\n$/, '');
    return extracted;
  }, [children]);

  // Determine if this is a code block (has language class and not inline)
  const hasLanguage = className && className.includes('language-');
  const isCodeBlock = hasLanguage && inline !== true;

  const handleCopy = async () => {
    await navigator.clipboard.writeText(codeString);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Inline code - render as inline if explicitly inline or no language class
  if (!isCodeBlock || inline) {
    return (
      <code className="px-1.5 py-0.5 bg-pink-50 text-pink-600 rounded text-sm font-mono">
        {children}
      </code>
    );
  }

  // Code block with syntax highlighting
  // Render SyntaxHighlighter directly - it will replace the pre+code structure
  return (
    <div className="relative group my-4 -mx-4 sm:mx-0 overflow-hidden">
      <div className="absolute top-3 right-3 z-10">
        <button
          onClick={handleCopy}
          className="flex items-center gap-1.5 px-2.5 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-200 text-xs rounded-md transition-colors opacity-0 group-hover:opacity-100 shadow-lg"
          title="Copy code"
        >
          {copied ? (
            <>
              <Check size={14} />
              <span>Copied!</span>
            </>
          ) : (
            <>
              <Copy size={14} />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>
      <div className="overflow-x-auto">
        <SyntaxHighlighter
          style={vscDarkPlus}
          language={language || 'text'}
          customStyle={{
            margin: 0,
            borderRadius: '0.5rem',
            padding: '1rem',
            paddingTop: '1.5rem',
            fontSize: '0.875rem',
            lineHeight: '1.5',
            overflow: 'visible',
          }}
          codeTagProps={{
            style: {
              fontSize: '0.875rem',
              fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
            }
          }}
        >
          {codeString}
        </SyntaxHighlighter>
      </div>
    </div>
  );
};

// =====================================================
// CHAT MESSAGE COMPONENT
// =====================================================

export interface ChatMessageProps {
  messageId: string;
  onAnnotationClick?: (annotation: AnnotationReference) => void;
}

export const ChatMessage = React.memo(function ChatMessage({
  messageId,
  onAnnotationClick
}: ChatMessageProps) {
  // Read directly from store - component only re-renders when THIS message changes
  const message = useChatStore((state) =>
    state.messages.find(m => m.id === messageId)
  );

  // Hooks must be called before any early return
  // Memoize processed content to avoid re-processing on parent re-renders
  const processedContent = useMemo(() =>
    message ? preprocessMathContent(message.content) : '',
    [message]
  );

  const handleAnnotationClick = useCallback((annotation: AnnotationReference) => {
    onAnnotationClick?.(annotation);
  }, [onAnnotationClick]);

  // Early return after hooks
  if (!message) return null;

  return (
    <div
      className={`flex gap-4 ${
        message.role === 'user' ? 'justify-end' : 'justify-start'
      } animate-in fade-in slide-in-from-bottom-2 duration-300`}
    >
      {/* Avatar for Assistant */}
      {message.role !== 'user' && (
        <div className="w-8 h-8 rounded-lg bg-white border border-gray-100 flex-shrink-0 flex items-center justify-center shadow-sm mt-1">
          <Bot size={16} className="text-indigo-600" />
        </div>
      )}

      <div
        className={`max-w-[85%] sm:max-w-[75%] rounded-2xl shadow-sm text-sm leading-relaxed ${
          message.role === 'user'
          ? 'bg-gradient-to-br from-blue-600 to-blue-700 text-white rounded-tr-sm p-4'
          : 'bg-white border border-gray-100 text-slate-700 rounded-tl-sm overflow-visible'
        }`}
      >
        {message.role === 'user' ? (
          message.content
        ) : (
          <>
            <div className="p-4 markdown-content prose prose-sm max-w-none prose-headings:text-slate-800 prose-p:text-slate-700 prose-a:text-blue-600 hover:prose-a:underline prose-strong:text-slate-900 prose-code:text-pink-600 prose-code:bg-pink-50 prose-code:px-1 prose-code:rounded prose-code:before:content-none prose-code:after:content-none prose-pre:bg-transparent prose-pre:p-0 prose-pre:m-0 prose-pre:border-0 overflow-x-auto">
              <ReactMarkdown
                remarkPlugins={[
                  remarkGfm,
                  [remarkMath, { singleDollarTextMath: true }]
                ]}
                rehypePlugins={[
                  [rehypeKatex, {
                    strict: false,
                    trust: true,
                    fleqn: false
                  }]
                ]}
                components={{
                  code: CodeBlock,
                  pre: ({ children }: { children?: React.ReactNode }) => {
                    // ReactMarkdown wraps code blocks in <pre><code>
                    // CodeBlock component will render SyntaxHighlighter which replaces both pre and code
                    // So we just pass through the code element - CodeBlock handles the rendering
                    return children as React.ReactElement;
                  },
                  ul: ({ children }: { children?: React.ReactNode }) => {
                    return <ul>{children}</ul>;
                  },
                  ol: ({ children }: { children?: React.ReactNode }) => {
                    return <ol>{children}</ol>;
                  },
                  li: ({ children }: { children?: React.ReactNode }) => {
                    return <li>{children}</li>;
                  },
                }}
              >
                {processedContent}
              </ReactMarkdown>
            </div>

            {/* Annotation References */}
            {message.annotations && message.annotations.length > 0 && (
              <div className="border-t border-gray-100 bg-gradient-to-r from-yellow-50 to-amber-50 p-3">
                <div className="flex items-center gap-2 text-xs font-medium text-amber-700 mb-2">
                  <FileText size={14} />
                  <span>Referenced in PDF</span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {message.annotations
                    .filter((annotation): annotation is NonNullable<typeof annotation> =>
                      annotation != null && typeof annotation.pageNumber === 'number'
                    )
                    .map((annotation, idx) => (
                    <button
                      key={`${message.id}-${annotation.pageNumber}-${idx}-${annotation.sourceImageUrl || 'no-img'}`}
                      onClick={() => handleAnnotationClick(annotation)}
                      className="group flex items-center gap-2 px-3 py-1.5 bg-white border border-amber-200 rounded-lg text-xs text-slate-700 hover:bg-amber-100 hover:border-amber-300 hover:text-amber-800 transition-all shadow-sm hover:shadow"
                    >
                      {annotation.sourceImageUrl && (
                        <span className="h-6 w-6 rounded border border-amber-200 overflow-hidden bg-white flex-shrink-0">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={annotation.sourceImageUrl}
                            alt="Annotation preview"
                            className="h-full w-full object-cover"
                            loading="lazy"
                          />
                        </span>
                      )}
                      <span className="font-semibold text-amber-600">
                        Page {annotation.pageNumber}
                      </span>
                      {annotation.explanation && (
                        <>
                          <span className="text-gray-300">|</span>
                          <span className="max-w-[150px] truncate text-slate-500 group-hover:text-slate-700">
                            {annotation.explanation}
                          </span>
                        </>
                      )}
                      <ArrowRight size={12} className="text-amber-500 group-hover:translate-x-0.5 transition-transform" />
                    </button>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {/* Avatar for User */}
      {message.role === 'user' && (
        <div className="w-8 h-8 rounded-lg bg-blue-100 flex-shrink-0 flex items-center justify-center mt-1">
          <User size={16} className="text-blue-600" />
        </div>
      )}
    </div>
  );
});
