import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import { MagnifyingGlassIcon, ArrowPathIcon, DocumentTextIcon } from '@heroicons/react/24/outline';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: string[];
}

interface TransactionResult {
  transaction_id: string;
  analysis: string;
  logs: any[];
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    { 
      role: 'assistant', 
      content: 'Xin chào! Tôi là AI Logs Analyst. Bạn có thể hỏi về logs hoặc tìm kiếm giao dịch theo transID.' 
    }
  ]);
  
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);
  const [transactionId, setTransactionId] = useState('');
  const [transactionResult, setTransactionResult] = useState<TransactionResult | null>(null);
  
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!input.trim()) return;
    
    const userMessage: Message = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    
    try {
      const response = await axios.post(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080'}/query`, {
        query: input,
        time_range: '24h'
      });
      
      const botMessage: Message = { 
        role: 'assistant', 
        content: response.data.response,
        sources: response.data.sources
      };
      
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error('Error querying logs:', error);
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Xin lỗi, tôi gặp lỗi khi xử lý câu hỏi của bạn.' }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSyncLogs = async () => {
    setIsSyncing(true);
    try {
      const response = await axios.post(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080'}/sync-logs`, {
        index_pattern: 'logstash-*',
        hours_back: 24
      });
      
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `Logs đã được đồng bộ thành công! ${response.data.logs_added} logs được thêm vào.` }
      ]);
    } catch (error) {
      console.error('Error syncing logs:', error);
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Xin lỗi, tôi gặp lỗi khi đồng bộ logs.' }
      ]);
    } finally {
      setIsSyncing(false);
    }
  };

  const handleTransactionSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!transactionId.trim()) return;
    
    setMessages((prev) => [...prev, { role: 'user', content: `Tìm kiếm giao dịch: ${transactionId}` }]);
    setIsLoading(true);
    
    try {
      const response = await axios.post(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080'}/transaction`, {
        transaction_id: transactionId
      });
      
      setTransactionResult(response.data);
      
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: response.data.analysis }
      ]);
    } catch (error) {
      console.error('Error searching transaction:', error);
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Xin lỗi, tôi gặp lỗi khi tìm kiếm giao dịch.' }
      ]);
    } finally {
      setIsLoading(false);
      setTransactionId('');
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      <header className="bg-blue-600 p-4 text-white">
        <h1 className="text-2xl font-bold">AI Logs Analysis System</h1>
      </header>
      
      <div className="flex flex-1 overflow-hidden p-4">
        {/* Main chat area */}
        <div className="flex flex-col flex-1 bg-white rounded-lg shadow overflow-hidden mr-4">
          <div className="flex-1 p-4 overflow-y-auto">
            {messages.map((message, index) => (
              <div 
                key={index} 
                className={`mb-4 ${message.role === 'user' ? 'text-right' : 'text-left'}`}
              >
                <div 
                  className={`inline-block p-3 rounded-lg ${
                    message.role === 'user' 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-gray-100 text-gray-800'
                  } max-w-[80%]`}
                >
                  <ReactMarkdown>{message.content}</ReactMarkdown>
                </div>
                
                {message.sources && (
                  <div className="mt-2 text-left text-xs text-gray-500 max-w-[80%] ml-auto">
                    <details>
                      <summary className="cursor-pointer">Nguồn logs</summary>
                      <pre className="mt-1 p-2 bg-gray-100 rounded overflow-x-auto text-xs">
                        {message.sources.map((source, idx) => (
                          <div key={idx} className="mb-2">
                            {typeof source === 'string' ? source : JSON.stringify(source, null, 2)}
                          </div>
                        ))}
                      </pre>
                    </details>
                  </div>
                )}
              </div>
            ))}
            {isLoading && (
              <div className="text-center p-4">
                <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-blue-600 border-r-transparent"></div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
          
          <form onSubmit={handleSubmit} className="border-t p-4">
            <div className="flex">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Nhập câu hỏi về logs..."
                className="flex-1 p-2 border rounded-l-lg focus:outline-none focus:ring-2 focus:ring-blue-600"
                disabled={isLoading}
              />
              <button
                type="submit"
                className="bg-blue-600 text-white p-2 rounded-r-lg hover:bg-blue-700 disabled:bg-blue-300"
                disabled={isLoading || !input.trim()}
              >
                <MagnifyingGlassIcon className="h-6 w-6" />
              </button>
            </div>
          </form>
        </div>
        
        {/* Right sidebar */}
        <div className="w-80 bg-white rounded-lg shadow overflow-hidden flex flex-col">
          <div className="p-4 bg-gray-50 border-b">
            <h2 className="text-lg font-semibold">Công cụ</h2>
          </div>
          
          <div className="p-4 space-y-4">
            {/* Transaction search */}
            <div>
              <h3 className="font-medium mb-2">Tìm kiếm giao dịch</h3>
              <form onSubmit={handleTransactionSearch} className="flex">
                <input
                  type="text"
                  value={transactionId}
                  onChange={(e) => setTransactionId(e.target.value)}
                  placeholder="Nhập TransID..."
                  className="flex-1 p-2 text-sm border rounded-l-lg focus:outline-none focus:ring-2 focus:ring-blue-600"
                  disabled={isLoading}
                />
                <button
                  type="submit"
                  className="bg-blue-600 text-white p-2 rounded-r-lg hover:bg-blue-700 disabled:bg-blue-300"
                  disabled={isLoading || !transactionId.trim()}
                >
                  <DocumentTextIcon className="h-5 w-5" />
                </button>
              </form>
            </div>
            
            {/* Sync logs button */}
            <div>
              <h3 className="font-medium mb-2">Cập nhật vector database</h3>
              <button
                onClick={handleSyncLogs}
                className="w-full flex items-center justify-center bg-green-600 text-white p-2 rounded-lg hover:bg-green-700 disabled:bg-green-300"
                disabled={isSyncing}
              >
                <ArrowPathIcon className={`h-5 w-5 mr-2 ${isSyncing ? 'animate-spin' : ''}`} />
                {isSyncing ? 'Đang đồng bộ...' : 'Đồng bộ logs mới'}
              </button>
            </div>
            
            {/* System health status */}
            <div className="mt-4">
              <h3 className="font-medium mb-2">Trạng thái hệ thống</h3>
              <div className="p-3 bg-gray-50 rounded-lg text-sm">
                <StatusIndicator />
              </div>
            </div>
            
            {/* Predefined queries */}
            <div>
              <h3 className="font-medium mb-2">Câu hỏi nhanh</h3>
              <div className="space-y-2">
                <button
                  onClick={() => {
                    setInput("Tìm tất cả các lỗi trong hệ thống trong 6 giờ qua");
                  }}
                  className="w-full text-left p-2 text-sm bg-gray-50 hover:bg-gray-100 rounded"
                >
                  Tìm lỗi trong 6 giờ qua
                </button>
                <button
                  onClick={() => {
                    setInput("Thống kê số lượng giao dịch thành công và thất bại hôm nay");
                  }}
                  className="w-full text-left p-2 text-sm bg-gray-50 hover:bg-gray-100 rounded"
                >
                  Thống kê giao dịch hôm nay
                </button>
                <button
                  onClick={() => {
                    setInput("Phân tích nguyên nhân thất bại phổ biến nhất");
                  }}
                  className="w-full text-left p-2 text-sm bg-gray-50 hover:bg-gray-100 rounded"
                >
                  Nguyên nhân thất bại phổ biến
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function StatusIndicator() {
  const [status, setStatus] = useState<any>({
    api: 'checking',
    elasticsearch: 'checking',
    chroma: 'checking',
    ollama: 'checking'
  });

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const response = await axios.get(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080'}/health`);
        setStatus(response.data);
      } catch (error) {
        console.error('Error checking system status:', error);
        setStatus({
          api: 'error',
          elasticsearch: 'error',
          chroma: 'error',
          ollama: 'error'
        });
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 30000); // Check every 30 seconds
    
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    if (status === 'healthy' || status === 'green') return 'bg-green-500';
    if (status === 'yellow') return 'bg-yellow-500';
    if (status === 'checking') return 'bg-blue-500 animate-pulse';
    return 'bg-red-500';
  };

  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center">
        <span>API:</span>
        <span className={`w-3 h-3 rounded-full ${getStatusColor(status.api)}`}></span>
      </div>
      <div className="flex justify-between items-center">
        <span>Elasticsearch:</span>
        <span className={`w-3 h-3 rounded-full ${getStatusColor(status.elasticsearch)}`}></span>
      </div>
      <div className="flex justify-between items-center">
        <span>ChromaDB:</span>
        <span className={`w-3 h-3 rounded-full ${getStatusColor(status.chroma)}`}></span>
      </div>
      <div className="flex justify-between items-center">
        <span>Ollama (LLM):</span>
        <span className={`w-3 h-3 rounded-full ${getStatusColor(status.ollama)}`}></span>
      </div>
    </div>
  );
}
