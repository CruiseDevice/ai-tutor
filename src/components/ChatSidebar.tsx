import { ChevronLeft, ChevronRight, MessageSquare } from "lucide-react";
import { useEffect, useState } from "react";

interface Conversation {
  id: string;
  documentId: string;
  title: string;  // document title
  updatedAt: string
}

interface ChatSidebarProps {
  userId: string;
  onSelectConversation: (conversationId: string, documentId: string) => void;
  currentConversationId: string | null;
}

export default function ChatSidebar({
  userId,
  onSelectConversation,
  currentConversationId,
}: ChatSidebarProps) {
  const [isOpen, setIsOpen] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [conversations, setConversations] = useState<Conversation[]>([]);

  useEffect(() => {
    if (!userId) return;

    const fetchConversations = async () => {
      setIsLoading(true);
      try {
        const response = await fetch('/api/conversations');
        if (!response.ok) {
          throw new Error('Failed to fetch conversations');
        }
        
        const data = await response.json();
        setConversations(data.conversations);
      } catch (error) {
        console.error('Error fetching conversations: ', error)
      } finally {
        setIsLoading(false);
      }
    };

    fetchConversations();
  }, [userId]);
  return (
    <div 
      className={`h-full bg-gray-100 border-r border-gray-200 transition-all duration-300 flex flex-col ${
        isOpen ? 'w-64' : 'w-12'
      }`}
    >
      {/* toggle button */}
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="p-2 self-end text-gray-500 hover:text-gray-700"
      >
        {isOpen ? <ChevronLeft size={20} /> : <ChevronRight size={20} />}
      </button>

      {/* sidebar content - only shown when open */}
      {isOpen && (
        <div className="p-4 flex-1 overflow-y-auto">
          <h2 className="font-semibold text-lg mb-4">Chat History</h2>
          {isLoading ? (
            <div className="text-gray-500">Loading...</div>
          ) : conversations.length === 0 ? (
            <div className="text-gray-500">No conversations yet</div>
          ) : (
            <ul className="space-y-2">
              {conversations.map((conversation) => (
                <li key={conversation.id}>
                  <button
                    onClick={() => onSelectConversation(conversation.id, conversation.documentId)}
                    className={`w-full text-left p-2 rounded flex items-center gap-2 hover:bg-gray-200 ${currentConversationId == conversation.id ? 'bg-blue-100' : ''}`}
                  >
                    <MessageSquare size={16}/>
                    <div className="overflow-hidden">
                      <div className="truncate text-sm font-medium">
                        {conversation.title}
                      </div>
                      <div className="text-xs text-gray-500">
                        {new Date(conversation.updatedAt).toLocaleDateString()}
                      </div>
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}