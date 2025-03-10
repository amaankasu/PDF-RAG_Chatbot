import React, { useState, useRef } from 'react';
import { GoLaw } from 'react-icons/go';
import './App.css';

// ReasoningDropdown component (Claude's UI style)
const ReasoningDropdown = ({ reasoningTime, finalContext }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="reasoning-container">
      <div 
        className="reasoning-header"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        {`Reasoned for ${reasoningTime} seconds`}
        <span className={`dropdown-icon ${isExpanded ? 'rotated' : ''}`}>
          ▼
        </span>
      </div>
      {isExpanded && (
        <div className="reasoning-content">
          <pre>{finalContext}</pre>
        </div>
      )}
    </div>
  );
};

function ChatbotUI() {
  // Initialize with empty data for dynamic updates.
  const [messages, setMessages] = useState([]);
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [uploadedPdf, setUploadedPdf] = useState(null);
  
  const fileInputRef = useRef(null);

  // Fixed backend URL for advanced model
  const backendUrl = 'http://127.0.0.1:5000';

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (file && file.type === 'application/pdf') {
      const displayName = file.name.length > 30 ? file.name.substring(0, 27) + '...' : file.name;
      setUploadedPdf({
        name: displayName,
        type: 'PDF'
      });
      // Create a FormData to send the file
      const formData = new FormData();
      formData.append('file', file);
      try {
        const response = await fetch(`${backendUrl}/upload`, {
          method: 'POST',
          body: formData,
        });
        const data = await response.json();
        if (!response.ok) {
          console.error(data.error);
        } else {
          console.log(data.message);
        }
      } catch (err) {
        console.error("Error uploading PDF:", err);
      }
    }
  };

  const handleSendMessage = async () => {
    if (currentQuestion.trim() === '') return;
    
    const queryText = currentQuestion;
    
    // Add user message
    const newUserMessage = {
      id: messages.length + 1,
      type: 'user',
      content: queryText,
    };
    setMessages([...messages, newUserMessage]);
    
    // Add a placeholder bot message with default "Analyzing" values
    const placeholderBotMessage = {
      id: messages.length + 2,
      type: 'bot',
      reasoningTime: 'Analyzing',
      content: 'Analyzing',
      finalContext: '',
    };
    setMessages(prevMessages => [...prevMessages, placeholderBotMessage]);
    setCurrentQuestion('');

    try {
      const response = await fetch(`${backendUrl}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        // Send the query with model set to advanced (even if backend ignores it)
        body: JSON.stringify({ query: queryText, model: "advanced" }),
      });
      const data = await response.json();
      if (!response.ok) {
        console.error(data.error);
        // Update placeholder with error
        setMessages(prevMessages =>
          prevMessages.map(msg =>
            msg.id === placeholderBotMessage.id 
              ? { ...msg, content: data.error, reasoningTime: 0 }
              : msg
          )
        );
      } else {
        // Update placeholder with final answer, reasoning time, and final context
        setMessages(prevMessages =>
          prevMessages.map(msg =>
            msg.id === placeholderBotMessage.id 
              ? { 
                  ...msg, 
                  content: data.answer, 
                  reasoningTime: data.reasoningTime,
                  finalContext: data.finalContext 
                }
              : msg
          )
        );
      }
    } catch (err) {
      console.error("Error querying:", err);
      setMessages(prevMessages =>
        prevMessages.map(msg =>
          msg.id === placeholderBotMessage.id 
            ? { ...msg, content: "Error querying", reasoningTime: 0 }
            : msg
        )
      );
    }
  };
  
  const triggerFileInput = () => {
    fileInputRef.current.click();
  };

  return (
    <div className="chatbot-container">
      {/* Header */}
      <header className="chatbot-header">
        <div className="header-left">
          <div className="logo-container">
            <GoLaw size={32} color="#000" style={{ cursor: 'pointer' }} />
          </div>
          <div className="chatgpt-title">
            Legal RAG - Your Law Firm’s Private, Reliable AI Assistant.
          </div>
        </div>
        <div className="header-right">
          <button className="share-button">
            <span className="share-icon"></span>
            Share
          </button>
          <div className="user-avatar">a</div>
        </div>
      </header>

      {/* Chat Content */}
      <div className="chat-content">
        {uploadedPdf && (
          <div className="pdf-attachment">
            <div className="pdf-icon"></div>
            <div className="pdf-details">
              <div className="pdf-name">{uploadedPdf.name}</div>
              <div className="pdf-type">{uploadedPdf.type}</div>
            </div>
          </div>
        )}
        
        {messages.map((message) => (
          <div key={message.id} className={`message-container ${message.type === 'user' ? 'user-message-container' : ''}`}>
            {message.type === 'user' ? (
              <div className="user-message">{message.content}</div>
            ) : (
              <div className="bot-message-container">
                {message.reasoningTime === 'Analyzing' ? (
                  <div className="reasoning-time">Analyzing</div>
                ) : (
                  // Render the ReasoningDropdown if finalContext exists
                  message.finalContext ? (
                    <ReasoningDropdown 
                      reasoningTime={message.reasoningTime}
                      finalContext={message.finalContext}
                    />
                  ) : (
                    <div className="reasoning-time">{`Reasoned for ${message.reasoningTime} seconds`}</div>
                  )
                )}
                <div className="bot-message">{message.content}</div>
                <div className="message-actions">
                  <button className="action-button copy-button"></button>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Input Area */}
      <div className="input-area">
        <div className="input-container">
          <input
            type="text"
            className="message-input"
            placeholder="Enter your Query"
            value={currentQuestion}
            onChange={(e) => setCurrentQuestion(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
          />
          
          <div className="input-buttons">
            <button className="attachment-button" onClick={triggerFileInput}>
              <span className="plus-icon"></span>
            </button>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              accept="application/pdf"
              style={{ display: 'none' }}
            />
            <button className="send-button" onClick={handleSendMessage}>
              <span className="send-icon"></span>
            </button>
          </div>
        </div>
      </div>
      
      <button className="scroll-bottom-button">
        <span className="arrow-down-icon"></span>
      </button>
    </div>
  );
}

export default ChatbotUI;
