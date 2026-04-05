import React, { useState, useRef } from 'react';
import { PlusCircle, Send, FileText, Link2, X, Image as ImageIcon, Loader2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown'; 

const promptChips = [
  { id: "pc-1", icon: FileText, text: "Rank Documents (Search Retrieval)" },
  { id: "pc-2", icon: FileText, text: "Check Plagiarism & Integrity" },
  { id: "pc-3", icon: FileText, text: "Cluster by Theme / Category" },
  { id: "pc-4", icon: Link2, text: "Compare 10-K Annual Reports" },
];

const SUPPORTED_TYPES = new Set([
  'application/pdf', 
  'application/msword', 
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 
  'application/vnd.ms-excel', 
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 
  'text/csv', 
  'image/jpeg',
  'image/png',
  'image/jpg'
]);

function ChatWorkspace() {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState("");
  const [attachments, setAttachments] = useState([]);
  const [isMenuOpen, setIsMenuOpen] = useState(false); 
  const [errorMsg, setErrorMsg] = useState(null); 
  const [isTyping, setIsTyping] = useState(false); 
  const [agentStatus, setAgentStatus] = useState("Agent is thinking..."); 
  
  const fileInputRef = useRef(null);
  const abortControllerRef = useRef(null);

  const handleMenuClick = (fileTypeStr) => {
    setIsMenuOpen(false); 
    if (fileInputRef.current) {
      fileInputRef.current.accept = fileTypeStr;
      fileInputRef.current.click();
    }
  };

  const processFiles = (files) => {
    const validFiles = [];
    let hasError = false;

    files.forEach(file => {
      if (SUPPORTED_TYPES.has(file.type)) {
        validFiles.push({
          originalFile: file, 
          name: file.name,
          type: file.type || '', 
          id: `file-${Date.now()}-${Math.random()}`
        });
      } else {
        hasError = true;
      }
    });

    if (validFiles.length > 0) {
      setAttachments((prev) => [...prev, ...validFiles]);
    }

    if (hasError) {
      setErrorMsg("Unsupported file type. Please use PDF, Word, Excel, CSV, or Images.");
      setTimeout(() => setErrorMsg(null), 3000); 
    }
  };

  const handleFileChange = (e) => {
    const files = Array.from(e.target.files);
    if (files.length > 0) processFiles(files);
    e.target.value = null; 
  };

  const handlePaste = (e) => {
    const pastedFiles = Array.from(e.clipboardData.files);
    if (pastedFiles.length > 0) {
      e.preventDefault(); 
      processFiles(pastedFiles); 
    }
  };

  const removeAttachment = (idToRemove) => {
    setAttachments(attachments.filter((file) => file.id !== idToRemove));
  };

  const handleCancelRequest = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort(); 
    }
  };

  const handleSendMessage = async () => {
    if (!inputText.trim() && attachments.length === 0) return;

    const newMsg = { 
      id: `msg-${Date.now()}`,
      role: 'user', 
      content: inputText,
      files: [...attachments] 
    };
    
    // Extract history before adding the new message so backend gets proper context
    const historyToPass = messages.map(m => ({ role: m.role, content: m.content }));
    
    setMessages((prev) => [...prev, newMsg]);
    setInputText("");
    setAttachments([]); 
    setIsTyping(true); 
    setAgentStatus("Connecting to Orchestrator...");

    abortControllerRef.current = new AbortController();

    const formData = new FormData();
    formData.append("prompt", newMsg.content);
    formData.append("history", JSON.stringify(historyToPass)); // Added memory payload
    
    newMsg.files.forEach((fileObj) => {
      formData.append("documents", fileObj.originalFile); 
    });

    const botMsgId = `msg-${Date.now() + 1}`;
    setMessages((prev) => [...prev, { 
      id: botMsgId,
      role: 'assistant', 
      content: "" // Starts empty until the final result arrives
    }]);

    try {
      const response = await fetch("http://localhost:8001/api/analyze", {
        method: "POST",
        body: formData, 
        signal: abortControllerRef.current.signal 
      });

      if (!response.ok) throw new Error("Backend connection failed.");

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop(); // Keep any incomplete JSON chunk in the buffer

        for (const line of lines) {
          if (line.trim()) {
            try {
              const parsed = JSON.parse(line);
              if (parsed.type === "status") {
                setAgentStatus(parsed.message); // Update UI spinner text
              } else if (parsed.type === "result" || parsed.type === "error") {
                // Populate the chat bubble when the final response arrives
                setMessages((prev) => prev.map(msg => 
                  msg.id === botMsgId ? { ...msg, content: parsed.message } : msg
                ));
              }
            } catch (e) {
              console.error("Failed to parse stream event:", e, line);
            }
          }
        }
      }

      // Handle any remaining buffer
      if (buffer.trim()) {
        try {
          const parsed = JSON.parse(buffer);
          if (parsed.type === "result" || parsed.type === "error") {
            setMessages((prev) => prev.map(msg => 
              msg.id === botMsgId ? { ...msg, content: parsed.message } : msg
            ));
          }
        } catch (e) { }
      }

    } catch (error) {
      if (error.name === 'AbortError') {
        setMessages((prev) => prev.map(msg => 
            msg.id === botMsgId ? { ...msg, content: "🛑 **Cancelled:** The generation was stopped by the user." } : msg
        ));
      } else {
        console.error(error);
        setMessages((prev) => prev.map(msg => 
            msg.id === botMsgId ? { ...msg, content: "⚠️ **System Offline:** The Python backend is currently unreachable. Please make sure Service B/C is running." } : msg
        ));
      }
    } finally {
      setIsTyping(false); 
      abortControllerRef.current = null; 
    }
  };

  const getFileIcon = (fileType) => {
    if (fileType?.includes('image')) return <ImageIcon className="w-4 h-4 text-orange-400" />;
    return <FileText className="w-4 h-4 text-blue-400" />; 
  };

  return (
    <div className="flex flex-col h-screen w-screen bg-[#10141d] text-gray-200 font-sans">
      
      <header className="flex items-center justify-between p-4 px-8 bg-[#0c1016] border-b border-gray-800 shadow-md">
        <h1 className="text-2xl font-black tracking-tighter text-white flex items-center gap-2">
          <div className="bg-blue-600 p-1.5 rounded-lg">
            <FileText className="w-5 h-5 text-white" />
          </div>
          DOCIFY
        </h1>
        <div className="flex items-center gap-2">
          <div className="w-9 h-9 rounded-full bg-gradient-to-tr from-blue-600 to-indigo-500 border-2 border-gray-800 shadow-lg" />
        </div>
      </header>

      <main className="flex-grow flex flex-col items-center p-6 space-y-10 overflow-y-auto">
        {messages.length === 0 ? (
          <>
            <div className="flex flex-col items-center text-center max-w-2xl space-y-4 animate-in fade-in zoom-in duration-500 mt-10">
              <div className="p-4 bg-gray-900/50 rounded-3xl border border-gray-800 shadow-2xl">
                <FileText className="w-12 h-12 text-blue-500 opacity-80" />
              </div>
              <h2 className="text-4xl font-extrabold text-white tracking-tight">
                Welcome to Docify.
              </h2>
              <p className="text-gray-500 text-lg max-w-md">
                Leverage TF-IDF and Cosine Similarity to rank relevance, detect plagiarism, and cluster documents independent of their length.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-2xl w-full">
              {promptChips.map((chip) => (
                <div key={chip.id} className="flex items-center gap-3 p-4 bg-[#1b212b] rounded-2xl border border-gray-800 shadow-sm opacity-70 cursor-default">
                  <div className="p-2 bg-gray-900 rounded-lg">
                    <chip.icon className="w-5 h-5 text-blue-400" />
                  </div>
                  <span className="text-gray-300 text-sm font-semibold">{chip.text}</span>
                </div>
              ))}
            </div>
          </>
        ) : (
          <div className="w-full max-w-3xl flex flex-col gap-6 pb-20">
            {messages.map((msg) => {
              // Hide empty assistant bubbles while generating
              if (msg.role === 'assistant' && !msg.content) return null;
              
              return (
                <div key={msg.id} className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'} animate-in fade-in slide-in-from-bottom-2`}>
                  
                  {msg.files && msg.files.length > 0 && (
                    <div className="flex flex-wrap gap-2 mb-2 justify-end">
                      {msg.files.map((file) => (
                        <div key={file.id} className="flex items-center gap-2 bg-[#1b212b] border border-gray-700 px-3 py-1.5 rounded-xl text-xs">
                          {getFileIcon(file.type)}
                          <span className="truncate max-w-[150px] text-gray-300">{file.name}</span>
                        </div>
                      ))}
                    </div>
                  )}

                  <div className={`px-5 py-3 rounded-2xl text-sm leading-relaxed shadow-sm max-w-[85%] ${
                    msg.role === 'user' 
                    ? 'bg-blue-600 text-white rounded-br-sm whitespace-pre-wrap' 
                    : 'bg-[#1b212b] text-gray-200 border border-gray-800 rounded-bl-sm prose prose-invert max-w-none'
                  }`}>
                    {msg.role === 'assistant' ? (
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    ) : (
                      msg.content
                    )}
                  </div>
                </div>
              );
            })}
            
            {isTyping && (
              <div className="flex flex-col items-start animate-in fade-in slide-in-from-bottom-2">
                <div className="px-4 py-2.5 bg-[#1b212b] border border-gray-800 rounded-2xl rounded-bl-sm flex items-center gap-3 text-blue-400 shadow-md">
                  <div className="bg-blue-600 p-1 rounded-md animate-pulse">
                    <FileText className="w-4 h-4 text-white" />
                  </div>
                  <span className="text-sm font-medium text-gray-300 animate-pulse">{agentStatus}</span>
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      <footer className="w-full flex justify-center p-6 bg-[#10141d] pt-0">
        <div className="w-full max-w-3xl flex flex-col gap-2 relative">
          
          {errorMsg && (
            <div className="absolute -top-12 left-0 right-0 flex justify-center animate-in fade-in slide-in-from-bottom-2">
              <div className="bg-red-500/90 text-white px-4 py-2 rounded-lg text-sm font-semibold shadow-lg">
                {errorMsg}
              </div>
            </div>
          )}

          {attachments.length > 0 && (
            <div className="flex flex-wrap gap-2 px-2 animate-in fade-in slide-in-from-bottom-2">
              {attachments.map((file) => (
                <div key={file.id} className="flex items-center gap-2 bg-[#1b212b] border border-gray-700 pl-3 pr-1 py-1.5 rounded-xl text-sm group shadow-lg">
                  {getFileIcon(file.type)}
                  <span className="truncate max-w-[120px] text-gray-300">{file.name}</span>
                  <button 
                    onClick={() => removeAttachment(file.id)}
                    className="p-1 hover:bg-gray-700 rounded-full text-gray-500 hover:text-red-400 transition-colors cursor-pointer"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
          )}

          {isMenuOpen && (
            <div className="absolute bottom-16 left-2 bg-[#1b212b] border border-gray-700 p-2 rounded-2xl shadow-2xl flex flex-col gap-1 w-48 animate-in fade-in slide-in-from-bottom-4 z-50">
              <button onClick={() => handleMenuClick('.pdf')} className="flex items-center gap-3 p-2.5 hover:bg-gray-800 rounded-xl text-sm transition-colors cursor-pointer text-gray-300">
                <FileText className="text-red-400 w-4 h-4" /> PDF Document
              </button>
              <button onClick={() => handleMenuClick('.doc,.docx')} className="flex items-center gap-3 p-2.5 hover:bg-gray-800 rounded-xl text-sm transition-colors cursor-pointer text-gray-300">
                <FileText className="text-blue-400 w-4 h-4" /> Word Document
              </button>
              <button onClick={() => handleMenuClick('.xls,.xlsx,.csv')} className="flex items-center gap-3 p-2.5 hover:bg-gray-800 rounded-xl text-sm transition-colors cursor-pointer text-gray-300">
                <FileText className="text-green-400 w-4 h-4" /> Spreadsheet
              </button>
              <button onClick={() => handleMenuClick('image/jpeg,image/png,image/jpg')} className="flex items-center gap-3 p-2.5 hover:bg-gray-800 rounded-xl text-sm transition-colors cursor-pointer text-gray-300">
                <ImageIcon className="text-orange-400 w-4 h-4" /> JPG / Image
              </button>
            </div>
          )}

          <div className="relative flex items-center gap-2 p-2 rounded-2xl bg-[#1b212b] border border-gray-800 focus-within:border-blue-500/50 shadow-2xl transition-all px-4">
            <input 
              type="file" 
              multiple 
              ref={fileInputRef} 
              onChange={handleFileChange}
              className="hidden" 
            />

            <button 
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className={`p-2 rounded-xl transition-all cursor-pointer ${isMenuOpen ? 'bg-gray-800 text-blue-400 rotate-45' : 'text-gray-500 hover:text-blue-400 hover:bg-gray-800'}`}
              title="Add Attachment"
            >
              <PlusCircle className="w-6 h-6" />
            </button>

            <textarea 
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault(); 
                  if (!isTyping && (inputText.trim() || attachments.length > 0)) {
                    handleSendMessage();
                  }
                }
              }}
              onPaste={handlePaste} 
              placeholder="Paste text, links, or attach files to analyze..."
              disabled={isTyping} 
              rows={1}
              className="flex-grow bg-transparent text-gray-100 placeholder-gray-600 outline-none text-md py-2 px-2 resize-none overflow-y-auto max-h-32 min-h-[40px] disabled:opacity-50"
            />

            <button 
              onClick={isTyping ? handleCancelRequest : handleSendMessage}
              disabled={!isTyping && !inputText.trim() && attachments.length === 0}
              className={`p-2.5 rounded-xl text-white shadow-lg active:scale-95 transition-all disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer ${
                isTyping 
                ? 'bg-red-600 hover:bg-red-500 shadow-red-900/20' 
                : 'bg-blue-600 hover:bg-blue-500 shadow-blue-900/20'
              }`}
            >
              {isTyping ? <X className="w-5 h-5" /> : <Send className="w-5 h-5" />}
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default ChatWorkspace;