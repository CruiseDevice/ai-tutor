// Types for chat messages and workflow steps

import type { AnnotationReference, AgentMetadata } from './annotations';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  annotations?: AnnotationReference[];
  metadata?: AgentMetadata;
}

export interface WorkflowStep {
  node: string;
  status: 'pending' | 'in_progress' | 'completed';
  data?: Record<string, unknown>;
}
