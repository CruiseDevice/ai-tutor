import { Mic, Send } from "lucide-react";
import { useState } from "react";

interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

interface ChatInterfaceProps {
  documentId: string;
  userId: string;
  onVoiceRecord: () => void;
}

export default function ChatInterface({
  documentId,
  userId,
  onVoiceRecord,
}: ChatInterfaceProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // const messagesEndRef = useRef<HTMLDivElement>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = {
      role: 'user' as const,
      content: inputMessage.trim()
    };

    // Add user message to chat
    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const payload = {
        messages: [...messages, userMessage],
        documentId,
        userId
      };
      console.log('Sending to API: ', payload);

      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error('Failed to send message');
      }
      
      const assistantMessage = await response.json();
      console.log(assistantMessage)
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Chat error:', error);
      // TODO: Show error to user
    } finally {
      setIsLoading(false)
    }

  }

  const handleVoiceRecord = () => {
    setIsRecording(!isRecording);
    onVoiceRecord();
  }
  return (
    <div className="w-1/2 flex flex-col bg-gray-50">
      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${
              message.role==='user' ? 'justify-end' : 'justify-start'
            }`}
            >
              <div 
                className={`max-w-[80%] p-3 rounded-lg ${
                  message.role === 'user'
                  ? 'bg-blue-500 text-white'
                  : 'bg-white border border-gray-200'
                }`}
                >
                  {message.content}
              </div>
          </div>
        ))}
      </div>
      {/* chat input */}
      <div className="border-t border-gray-200 p-4">
        <form
          onSubmit={handleSubmit}
          className="flex gap-2">
          <button 
            type="button"
            onClick={handleVoiceRecord}
            className={`p-2 rounded-full ${
              isRecording ? 'bg-red-500' : 'bg-gray-200'
            } hover:opacity-80`}>
              <Mic size={20} className={isRecording ? 'text-white' : 'text-gray-600'}/>
            </button>
            <input 
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder="Ask about the document..."
              className="flex-1 border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button
              type="submit"
              disabled={!inputMessage.trim() || isLoading}
              className="p-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send size={20}/>
            </button>
        </form>
      </div>
    </div>
  )
}