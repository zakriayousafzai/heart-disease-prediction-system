"use client";

import { JSX, useEffect, useState } from "react";
import {
  Chart as ChartJS,
  BarElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
} from "chart.js";
import { Bar } from "react-chartjs-2";

ChartJS.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend);

/* ---------- Types ---------- */
interface MetricsData {
  ann: number;
  rf: number;
  lr: number;
}

interface MetricCardProps {
  title: string;
  value: number;
  best: boolean;
}

/* ---------- Component ---------- */
export default function Metrics(): JSX.Element {
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [bestModel, setBestModel] = useState<string>("");

  useEffect(() => {
    fetch("http://localhost:5000/metrics")
      .then((res) => res.json())
      .then((data: MetricsData) => {
        setMetrics(data);

        // Find best model automatically
        const maxVal = Math.max(data.ann, data.rf, data.lr);
        const best =
          maxVal === data.ann
            ? "ANN"
            : maxVal === data.rf
            ? "Random Forest"
            : "Logistic Regression";

        setBestModel(best);
      });
  }, []);

  if (!metrics) {
    return <p className="text-center mt-10">Loading metrics...</p>;
  }

  const chartData = {
    labels: ["ANN", "Random Forest", "Logistic Regression"],
    datasets: [
      {
        label: "Accuracy (%)",
        data: [
          metrics.ann * 100,
          metrics.rf * 100,
          metrics.lr * 100,
        ],
        backgroundColor: ["#3b82f6", "#22c55e", "#f97316"],
      },
    ],
  };

  return (
    <div className="max-w-5xl mx-auto p-4">
      <h2 className="text-3xl font-bold text-center mb-6">
        Model Performance Metrics
      </h2>

      {/* 📱 Responsive Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
        <MetricCard
          title="ANN"
          value={metrics.ann}
          best={bestModel === "ANN"}
        />
        <MetricCard
          title="Random Forest"
          value={metrics.rf}
          best={bestModel === "Random Forest"}
        />
        <MetricCard
          title="Logistic Regression"
          value={metrics.lr}
          best={bestModel === "Logistic Regression"}
        />
      </div>

      {/* 📊 Bar Chart */}
      <div className="bg-white p-4 rounded shadow">
        <Bar data={chartData} />
      </div>
    </div>
  );
}

/* ---------- Metric Card ---------- */
function MetricCard({
  title,
  value,
  best,
}: MetricCardProps): JSX.Element {
  return (
    <div
      className={`p-4 rounded shadow text-center border-2 ${
        best ? "border-green-500 bg-green-50" : "border-gray-200"
      }`}
    >
      <h3 className="text-lg font-semibold">{title}</h3>
      <p className="text-2xl font-bold mt-2">
        {(value * 100).toFixed(2)}%
      </p>

      {best && (
        <span className="inline-block mt-2 text-green-600 font-semibold">
          🟢 Best Model
        </span>
      )}
    </div>
  );
}
