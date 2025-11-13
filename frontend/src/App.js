import React, { useState } from "react";
import "./App.css";

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setAnswer("");
    setSources([]);

    try {
      const response = await fetch("http://127.0.0.1:8000/query", {
    method: "POST",
    headers: {
        "Content-Type": "application/json",
    },
    body: JSON.stringify({ question }),
      });


      if (!response.ok) {
        throw new Error("Backend returned an error");
      }

      const data = await response.json();
      setAnswer(data.answer || "No answer found.");
      setSources(data.sources || []);
    } catch (err) {
      console.error(err);
      setError("❌ Failed to connect to backend. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <h1 className="title">RAG Assistant 🤖</h1>

      <form onSubmit={handleSubmit} className="input-form">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask your question..."
          className="input-box"
          required
        />
        <button type="submit" className="submit-btn">
          Ask
        </button>
      </form>

      {loading && <div className="loader">⏳ Thinking...</div>}

      {error && <div className="error">{error}</div>}

      {answer && !loading && (
        <div className="answer-box">
          <h3>Answer:</h3>
          <p>{answer}</p>
        </div>
      )}

      {sources.length > 0 && (
        <div className="sources-box">
          <h3>Sources:</h3>
          <ul>
            {sources.map((src, idx) => (
              <li key={idx}>{src}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;
