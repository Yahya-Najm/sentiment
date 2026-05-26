"use client";

import { useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

interface Prediction {
  label: "neg" | "pos";
  confidence: number;
  neg_prob: number;
  pos_prob: number;
}

export default function Home() {
  const [text, setText]       = useState("");
  const [result, setResult]   = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState<string | null>(null);

  async function analyze() {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ text }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail ?? `Server error ${res.status}`);
      }

      setResult(await res.json());
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  const isPositive = result?.label === "pos";

  return (
    <main className="min-h-screen bg-gray-950 text-gray-100 flex flex-col items-center justify-center px-4 py-16">
      <div className="w-full max-w-2xl space-y-8">

        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold tracking-tight">Sentiment Analysis</h1>
          <p className="text-gray-400 text-sm">
            DistilBERT fine-tuned on IMDB movie reviews
          </p>
        </div>

        {/* Input */}
        <div className="space-y-3">
          <textarea
            className="w-full h-44 rounded-xl bg-gray-900 border border-gray-700 focus:border-indigo-500 focus:outline-none p-4 text-sm resize-none placeholder-gray-600 transition"
            placeholder="Paste a movie review here…"
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <button
            onClick={analyze}
            disabled={loading || !text.trim()}
            className="w-full py-3 rounded-xl font-semibold text-sm bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed transition"
          >
            {loading ? "Analyzing…" : "Analyze"}
          </button>
        </div>

        {/* Error */}
        {error && (
          <p className="text-red-400 text-sm text-center">{error}</p>
        )}

        {/* Result card */}
        {result && (
          <div
            className={`rounded-xl border p-6 space-y-5 ${
              isPositive
                ? "bg-green-950 border-green-700"
                : "bg-red-950 border-red-700"
            }`}
          >
            <div className="flex items-center justify-between">
              <span
                className={`text-2xl font-bold ${
                  isPositive ? "text-green-400" : "text-red-400"
                }`}
              >
                {isPositive ? "Positive" : "Negative"}
              </span>
              <span className="text-gray-300 text-sm font-mono">
                {(result.confidence * 100).toFixed(1)}% confidence
              </span>
            </div>

            <div className="space-y-3">
              <ProbBar label="Positive" value={result.pos_prob} color="bg-green-500" />
              <ProbBar label="Negative" value={result.neg_prob} color="bg-red-500" />
            </div>
          </div>
        )}
      </div>
    </main>
  );
}

function ProbBar({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) {
  const pct = (value * 100).toFixed(1);
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs text-gray-400">
        <span>{label}</span>
        <span>{pct}%</span>
      </div>
      <div className="h-2 rounded-full bg-gray-800 overflow-hidden">
        <div
          className={`h-full rounded-full ${color} transition-all duration-500`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
