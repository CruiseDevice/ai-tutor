import { FileText } from "lucide-react";
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

const preprocessMathContent = (content: string): string => {
  const parts: Array<{ type: 'code' | 'text', content: string }> = [];
  const codeBlockRegex = /(```[\s\S]*?```|`[^`\n]+?`)/g;
  let lastIndex = 0;
  let match;

  while ((match = codeBlockRegex.exec(content)) !== null) {
    if (match.index > lastIndex) {
      parts.push({ type: 'text', content: content.slice(lastIndex, match.index) });
    }
    parts.push({ type: 'code', content: match[0] });
    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < content.length) {
    parts.push({ type: 'text', content: content.slice(lastIndex) });
  }

  return parts.map(part => {
    if (part.type === 'code') {
      return part.content;
    }

    let processed = part.content;
    processed = processed.replace(/\\\[([\s\S]*?)\\\]/g, (_match, inner) => `$$${inner}$$`);
    processed = processed.replace(/\\\((.*?)\\\)/g, (_match, inner) => `$${inner}$`);
    return processed;
  }).join('');
};

// =====================================================
// CODE BLOCK COMPONENT - Brutalist Terminal Style
// =====================================================

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

  const codeString = useMemo(() => {
    const extracted = React.Children.toArray(children)
      .map((child) => {
        if (typeof child === 'string') {
          return child;
        }
        if (React.isValidElement(child)) {
          const element = child as React.ReactElement<{ children?: React.ReactNode }>;
          if (element.props?.children) {
            return React.Children.toArray(element.props.children)
              .map(c => typeof c === 'string' ? c : String(c))
              .join('');
          }
        }
        return String(child);
      })
      .join('')
      .replace(/\n$/, '');
    return extracted;
  }, [children]);

  const hasLanguage = className && className.includes('language-');
  const isCodeBlock = hasLanguage && inline !== true;

  const handleCopy = async () => {
    await navigator.clipboard.writeText(codeString);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Inline code - brutalist style
  if (!isCodeBlock || inline) {
    return (
      <code className="brutalist-inline-code">
        {children}
      </code>
    );
  }

  // Code block - terminal style
  return (
    <div className="brutalist-code-block my-4 overflow-hidden">
      {/* Terminal header */}
      <div className="brutalist-code-header">
        <span className="opacity-75">[{language.toUpperCase() || 'TXT'}]</span>
        <button
          onClick={handleCopy}
          className="brutalist-copy-button"
        >
          {copied ? '[COPIED!]' : '[COPY]'}
        </button>
      </div>
      <div className="overflow-x-auto">
        <SyntaxHighlighter
          style={vscDarkPlus}
          language={language || 'text'}
          customStyle={{
            margin: 0,
            borderRadius: 0,
            padding: '1rem',
            fontSize: '0.8rem',
            lineHeight: '1.6',
            fontFamily: '"IBM Plex Mono", monospace',
            background: 'var(--ink)',
          }}
          codeTagProps={{
            style: {
              fontSize: '0.8rem',
              fontFamily: '"IBM Plex Mono", monospace',
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
// ANNOTATION COMPONENT - Brutalist Document Reference
// =====================================================

interface AnnotationPillProps {
  annotation: AnnotationReference;
  onClick: () => void;
  index: number;
}

const AnnotationPill = ({ annotation, onClick, index }: AnnotationPillProps) => {
  return (
    <button
      onClick={onClick}
      className="brutalist-annotation-pill flex items-center gap-2 px-3 py-2 min-h-[44px]"
      style={{ animationDelay: `${index * 50}ms` }}
    >
      {annotation.sourceImageUrl && (
        <span className="w-8 h-8 border-2 border-current overflow-hidden flex-shrink-0 bg-paper">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={annotation.sourceImageUrl}
            alt=""
            className="w-full h-full object-cover"
            loading="lazy"
          />
        </span>
      )}
      <span className="font-bold">
        P.{annotation.pageNumber.toString().padStart(2, '0')}
      </span>
      {annotation.explanation && (
        <span className="opacity-60 max-w-[120px] truncate">
          {annotation.explanation}
        </span>
      )}
      <span className="ml-auto">→</span>
    </button>
  );
};

// =====================================================
// BRUTALIST CHAT MESSAGE COMPONENT
// =====================================================

export interface ChatMessageProps {
  messageId: string;
  onAnnotationClick?: (annotation: AnnotationReference) => void;
}

export const ChatMessage = React.memo(function ChatMessage({
  messageId,
  onAnnotationClick
}: ChatMessageProps) {
  const message = useChatStore((state) =>
    state.messages.find(m => m.id === messageId)
  );

  const processedContent = useMemo(() =>
    message ? preprocessMathContent(message.content) : '',
    [message]
  );

  const handleAnnotationClick = useCallback((annotation: AnnotationReference) => {
    onAnnotationClick?.(annotation);
  }, [onAnnotationClick]);

  if (!message) return null;

  const isUser = message.role === 'user';
  const animationClass = isUser ? 'animate-[message-slide-in-user_0.3s_ease-out]' : 'animate-[message-slide-in-ai_0.3s_ease-out]';

  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'} ${animationClass}`}>
      {/* Avatar Badge */}
      <div className={`brutalist-avatar-${isUser ? 'user' : 'ai'} w-10 h-10 flex items-center justify-center flex-shrink-0`}>
        {isUser ? '[U]' : '[AI]'}
      </div>

      {/* Message Bubble */}
      <div className={`max-w-[85%] sm:max-w-[75%] ${isUser ? 'brutalist-message-user' : 'brutalist-message-ai'} pt-4 pb-3 px-4`}>
        {isUser ? (
          <p className="font-serif text-base leading-relaxed">
            {message.content}
          </p>
        ) : (
          <>
            {/* Markdown Content */}
            <div className="brutalist-prose prose-sm max-w-none">
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
                    return children as React.ReactElement;
                  },
                }}
              >
                {processedContent}
              </ReactMarkdown>
            </div>

            {/* Annotation References */}
            {message.annotations && message.annotations.length > 0 && (
              <div className="mt-4">
                <div className="brutalist-annotation-header flex items-center gap-2 mb-3">
                  <FileText size={14} />
                  <span>[SOURCE REFERENCES]</span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {message.annotations
                    .filter((annotation): annotation is NonNullable<typeof annotation> =>
                      annotation != null && typeof annotation.pageNumber === 'number'
                    )
                    .map((annotation, idx) => (
                      <AnnotationPill
                        key={`${message.id}-${annotation.pageNumber}-${idx}`}
                        annotation={annotation}
                        onClick={() => handleAnnotationClick(annotation)}
                        index={idx}
                      />
                    ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
});
