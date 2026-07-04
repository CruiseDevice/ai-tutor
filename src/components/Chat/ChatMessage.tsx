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

// ───────────────────────────────────────────────────────────────
// Inline citations (Phase 8 / Option A) — restyle the model's
// existing `[Page N]` markers as clickable superscripts keyed to the
// same filtered annotations the bibliography footer uses.
//
// The model already emits `[Page X]` per the system prompt; we just
// rewrite each one to a markdown link whose href encodes the
// annotation index (`#cite-<idx>`), which the custom `a` renderer
// turns into a `.citation` <sup>. No backend, no schema change.
//
// Code-block-aware: reuses the same split as preprocessMathContent so
// citations never appear inside code. `[Page N]` with no matching
// annotation stays literal (graceful fallback to the footer).
// ───────────────────────────────────────────────────────────────

const CITE_LINK_PREFIX = '#cite-';

const injectCitations = (
  content: string,
  validAnnotations: AnnotationReference[]
): string => {
  if (!validAnnotations.length) return content;

  // Map page number → 1-based footer index (matches BibliographyRow order).
  const pageToIndex = new Map<number, number>();
  validAnnotations.forEach((ann, idx) => {
    if (typeof ann.pageNumber === 'number' && !pageToIndex.has(ann.pageNumber)) {
      pageToIndex.set(ann.pageNumber, idx);
    }
  });

  // Split on code spans/blocks so we never touch citations inside code.
  const codeBlockRegex = /(```[\s\S]*?```|`[^`\n]+?`)/g;
  const parts: string[] = [];
  let lastIndex = 0;
  let match;

  while ((match = codeBlockRegex.exec(content)) !== null) {
    if (match.index > lastIndex) {
      parts.push(replacePageMarkers(content.slice(lastIndex, match.index), pageToIndex));
    }
    parts.push(match[0]); // code — unchanged
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < content.length) {
    parts.push(replacePageMarkers(content.slice(lastIndex), pageToIndex));
  }
  return parts.join('');
};

// Replace `[Page N]` (case-insensitive, any separator) with a citation
// link keyed to the annotation index. Unmatched pages are left as-is.
const replacePageMarkers = (
  text: string,
  pageToIndex: Map<number, number>
): string => {
  return text.replace(/\[\s*page\s+(\d+)\s*]/gi, (full, pageStr) => {
    const idx = pageToIndex.get(Number(pageStr));
    if (idx === undefined) return full; // no annotation for this page — leave literal
    const num = idx + 1;
    // Markdown link with the index as label and a sentinel href.
    return `[${num}](${CITE_LINK_PREFIX}${num})`;
  });
};


// =====================================================
// CODE BLOCK COMPONENT — dark ink panel with hairline header
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

  // Inline code — academic-prose already styles `code`, so render bare.
  if (!isCodeBlock || inline) {
    return (
      <code>
        {children}
      </code>
    );
  }

  // Code block — dark ink panel with a hairline header
  return (
    <div className="my-4 overflow-hidden rounded-sm" style={{ background: 'var(--ink)' }}>
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/10">
        <span className="font-mono text-[11px] tracking-wide text-white/60 uppercase">
          {language || 'text'}
        </span>
        <button
          onClick={handleCopy}
          className="font-mono text-[11px] text-white/70 hover:text-white transition-colors ease-desk"
        >
          {copied ? 'Copied' : 'Copy'}
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
            background: 'var(--ink)',
          }}
          codeTagProps={{
            style: {
              fontSize: '0.8rem',
              fontFamily: 'var(--font-mono)',
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
// BIBLIOGRAPHY ROW — academic source reference (Phase 2 / Option B)
// Replaces the brutalist bracket pill with a hanging-indent
// `N  Abstract, p. 3 — "quote…"` row keyed to message.annotations.
// Click → onAnnotationClick → store → PDF jump + mark-fade highlight.
// =====================================================

interface BibliographyRowProps {
  annotation: AnnotationReference;
  onClick: () => void;
  index: number;
}

const BibliographyRow = ({ annotation, onClick, index }: BibliographyRowProps) => {
  // Prefer the document's own source text as the quote; fall back to the
  // model's explanation, then to nothing (page-only citation).
  const quote = annotation.sourceText || annotation.explanation;
  const sourceLabel = annotation.explanation && annotation.sourceText
    ? annotation.explanation
    : null;

  return (
    <button
      onClick={onClick}
      className="bibliography-row w-full text-left"
      style={{ animationDelay: `${index * 50}ms` }}
    >
      <span className="bibliography-num">{index + 1}</span>
      <span className="min-w-0">
        {annotation.sourceImageUrl && (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={annotation.sourceImageUrl}
            alt=""
            className="w-9 h-9 object-cover rounded-xs float-right ml-2 mb-1 border border-hair"
            loading="lazy"
          />
        )}
        <span className="text-ink">
          {sourceLabel && <span className="text-subtle">{sourceLabel}, </span>}
          <span className="text-ink-2">p. {annotation.pageNumber}</span>
          {quote && (
            <>
              <span className="text-subtle"> — </span>
              <span className="text-ink-2 italic">&ldquo;{truncate(quote, 120)}&rdquo;</span>
            </>
          )}
        </span>
      </span>
    </button>
  );
};

const truncate = (s: string, n: number) =>
  s.length > n ? `${s.slice(0, n).trimEnd()}…` : s;

// =====================================================
// CHAT MESSAGE — academic correspondence (Phase 2)
// AI: avatar + label + plain prose on paper (no bubble).
// User: right-aligned, no avatar (correspondence feel).
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

  // Annotations with a valid page number — the same set the footer renders,
  // so inline superscripts and footer rows share one numbering.
  const validAnnotations = useMemo(
    () =>
      (message?.annotations ?? []).filter(
        (ann): ann is AnnotationReference =>
          ann != null && typeof ann.pageNumber === 'number'
      ),
    [message]
  );

  const processedContent = useMemo(() =>
    message ? injectCitations(preprocessMathContent(message.content), validAnnotations) : '',
    [message, validAnnotations]
  );

  const handleAnnotationClick = useCallback((annotation: AnnotationReference) => {
    onAnnotationClick?.(annotation);
  }, [onAnnotationClick]);

  // Resolve a citation-link href to its annotation, or undefined if not a citation.
  const citationFromHref = useCallback(
    (href: string | undefined): AnnotationReference | undefined => {
      if (!href || !href.startsWith(CITE_LINK_PREFIX)) return undefined;
      const idx = Number(href.slice(CITE_LINK_PREFIX.length)) - 1;
      return validAnnotations[idx];
    },
    [validAnnotations]
  );

  if (!message) return null;

  const isUser = message.role === 'user';
  const animationClass = isUser
    ? 'animate-[message-slide-in-user_0.3s_var(--ease)]'
    : 'animate-[message-slide-in-ai_0.3s_var(--ease)]';

  // USER — right-aligned, no avatar, desk-tinted bubble.
  if (isUser) {
    return (
      <div className={`flex justify-end ${animationClass}`}>
        <div
          className="max-w-[85%] sm:max-w-[75%] rounded-lg px-4 py-2.5"
          style={{ background: 'var(--desk)', borderTopRightRadius: 'var(--r-xs)', borderBottomRightRadius: 'var(--r-xs)' }}
        >
          <p className="font-serif text-base leading-relaxed text-ink">
            {message.content}
          </p>
        </div>
      </div>
    );
  }

  // AI — avatar + label + plain prose on paper, then optional sources.
  return (
    <div className={`flex gap-3 ${animationClass}`}>
      {/* Tutor avatar — accent gradient circle, single letter */}
      <div className="tutor-avatar w-9 h-9 flex-shrink-0 mt-1">
        A
      </div>

      <div className="min-w-0 flex-1">
        <div className="flex items-baseline gap-2 mb-1">
          <span className="font-serif font-semibold text-sm text-ink">AI Tutor</span>
        </div>

        {/* Prose on paper — no bubble */}
        <div className="academic-prose">
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
              a: ({ href, children }) => {
                const annotation = citationFromHref(href);
                if (annotation) {
                  return (
                    <sup
                      className="citation"
                      onClick={() => handleAnnotationClick(annotation)}
                    >
                      {children}
                    </sup>
                  );
                }
                // Normal link — preserve default behavior.
                return <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>;
              },
            }}
          >
            {processedContent}
          </ReactMarkdown>
        </div>

        {/* Sources — academic bibliography footer.
            Numbering shares validAnnotations with the inline superscripts
            above so a click on either jumps to the same PDF passage. */}
        {validAnnotations.length > 0 && (
          <div className="mt-5 pt-3 border-t border-hair-soft">
            <div className="font-mono text-[11px] uppercase tracking-wider text-faint mb-1.5">
              Sources
            </div>
            <div className="bibliography">
              {validAnnotations
                .map((annotation, idx) => (
                  <BibliographyRow
                    key={`${message.id}-${annotation.pageNumber}-${idx}`}
                    annotation={annotation}
                    onClick={() => handleAnnotationClick(annotation)}
                    index={idx}
                  />
                ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
});
