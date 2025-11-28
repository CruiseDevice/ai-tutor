import React from 'react';

interface WorkflowStep {
  node: string;
  status: 'pending' | 'in_progress' | 'completed';
  data?: any;
}

interface Props {
  steps: WorkflowStep[];
  visible: boolean;
}

const STEP_CONFIG = {
  understand_query: {
    label: 'Understanding Query',
    icon: '🔍',
    description: 'Analyzing query type and complexity',
  },
  retrieve_context: {
    label: 'Retrieving Context',
    icon: '📚',
    description: 'Finding relevant information',
  },
  generate_answer: {
    label: 'Generating Answer',
    icon: '✍️',
    description: 'Creating comprehensive response',
  },
  verify_response: {
    label: 'Verifying Quality',
    icon: '✅',
    description: 'Checking accuracy and citations',
  },
  format_response: {
    label: 'Formatting Response',
    icon: '📝',
    description: 'Finalizing answer',
  },
};

export default function AgentWorkflowProgress({ steps, visible }: Props) {
  if (!visible) return null;

  return (
    <div className="mb-4 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg shadow-sm">
      <div className="flex items-center gap-2 mb-3">
        <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
        <div className="text-sm font-semibold text-blue-900">
          Agent Workflow in Progress
        </div>
      </div>

      <div className="space-y-2">
        {steps.map((step, index) => {
          const config = STEP_CONFIG[step.node as keyof typeof STEP_CONFIG];
          if (!config) return null;

          return (
            <div
              key={step.node}
              className={`flex items-start gap-3 p-2 rounded transition-all ${
                step.status === 'completed' ? 'bg-white/50' :
                step.status === 'in_progress' ? 'bg-white shadow-sm' :
                'opacity-60'
              }`}
            >
              {/* Step indicator */}
              <div
                className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-semibold transition-all ${
                  step.status === 'completed'
                    ? 'bg-green-500 text-white'
                    : step.status === 'in_progress'
                    ? 'bg-blue-500 text-white animate-pulse'
                    : 'bg-gray-300 text-gray-600'
                }`}
              >
                {step.status === 'completed' ? '✓' : index + 1}
              </div>

              {/* Step content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-base">{config.icon}</span>
                  <span className="text-sm font-medium text-gray-900">
                    {config.label}
                  </span>
                  {step.status === 'in_progress' && (
                    <span className="text-xs text-blue-600 animate-pulse">●</span>
                  )}
                </div>

                <div className="text-xs text-gray-500 mt-0.5">
                  {config.description}
                </div>

                {/* Step metadata */}
                {step.data && step.status === 'completed' && (
                  <div className="mt-1 text-xs text-gray-600 space-y-0.5">
                    {step.node === 'understand_query' && step.data.query_type && (
                      <div>
                        <span className="font-medium">Type:</span> {step.data.query_type}
                        {step.data.complexity && (
                          <span className={`ml-2 px-1.5 py-0.5 rounded text-xs font-medium ${
                            step.data.complexity === 'complex' ? 'bg-red-100 text-red-700' :
                            step.data.complexity === 'moderate' ? 'bg-yellow-100 text-yellow-700' :
                            'bg-green-100 text-green-700'
                          }`}>
                            {step.data.complexity}
                          </span>
                        )}
                      </div>
                    )}
                    {step.node === 'retrieve_context' && step.data.chunks_retrieved && (
                      <div>
                        <span className="font-medium">Chunks:</span> {step.data.chunks_retrieved} retrieved
                        {step.data.chunks_used && ` → ${step.data.chunks_used} used`}
                      </div>
                    )}
                    {step.node === 'verify_response' && step.data.quality_score && (
                      <div>
                        <span className="font-medium">Quality:</span>{' '}
                        <span className={step.data.quality_score >= 8 ? 'text-green-600' : step.data.quality_score >= 7 ? 'text-yellow-600' : 'text-red-600'}>
                          {step.data.quality_score.toFixed(1)}/10
                        </span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
