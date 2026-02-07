"use client";

import { useState } from "react";
import axios from "axios";

interface RiskFactor {
  feature: string;
  value: string;
  impact: string;
  impact_score: number;
  direction: string;
  description: string;
}

interface Recommendation {
  category: string;
  icon: string;
  title: string;
  description: string;
  priority: string;
}

interface PredictionResult {
  ann_prediction: {
    result: string;
    probability: number;
  };
  rf_prediction: string;
  lr_prediction: string;
  id: number;
  risk_factors?: RiskFactor[];
  recommendations?: Recommendation[];
}

export default function PredictPage() {
  const [form, setForm] = useState({
    age: "",
    sex: "Male",
    chest_pain_type: "Typical Angina",
    resting_bp: "",
    cholesterol: "",
    fasting_bs: "<= 120 mg/dl",
    resting_ecg: "Normal",
    max_hr: "",
    exercise_angina: "No",
    oldpeak: "",
    st_slope: "Upsloping",
  });

  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await axios.post("http://localhost:5000/predict", {
        age: Number(form.age),
        sex: form.sex,
        chest_pain_type: form.chest_pain_type,
        resting_bp: Number(form.resting_bp),
        cholesterol: Number(form.cholesterol),
        fasting_bs: form.fasting_bs,
        resting_ecg: form.resting_ecg,
        max_hr: Number(form.max_hr),
        exercise_angina: form.exercise_angina,
        oldpeak: Number(form.oldpeak),
        st_slope: form.st_slope,
      });

      setResult(response.data);
    } catch {
      alert("❌ Backend error. Check Flask server.");
    } finally {
      setLoading(false);
    }
  };

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case "high":
        return "text-red-600";
      case "medium":
        return "text-yellow-600";
      case "low":
        return "text-green-600";
      default:
        return "text-gray-600";
    }
  };

  const getImpactIcon = (impact: string, direction: string) => {
    if (direction === "decreases risk") return "✅";
    switch (impact) {
      case "high":
        return "🔴";
      case "medium":
        return "🟡";
      case "low":
        return "🟢";
      default:
        return "⚪";
    }
  };

  const getPriorityBadgeColor = (priority: string) => {
    switch (priority) {
      case "high":
        return "bg-red-100 text-red-800 border-red-300";
      case "medium":
        return "bg-yellow-100 text-yellow-800 border-yellow-300";
      case "low":
        return "bg-green-100 text-green-800 border-green-300";
      default:
        return "bg-gray-100 text-gray-800 border-gray-300";
    }
  };

  const getRiskLevelColor = (result: string) => {
    if (result.includes("High")) return "bg-red-50 border-red-200";
    if (result.includes("Medium")) return "bg-yellow-50 border-yellow-200";
    return "bg-green-50 border-green-200";
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-red-50 to-gray-100 flex justify-center items-start p-4 py-8">
      <div className="bg-white w-full max-w-6xl rounded-2xl shadow-lg p-6 md:p-10">
        <h1 className="text-3xl font-bold text-center text-red-600 mb-2">
          🫀 Heart Disease Prediction
        </h1>
        <p className="text-center text-gray-600 mb-8">
          Please fill the form carefully. All fields are required.
        </p>

        <form
          onSubmit={handleSubmit}
          className="grid grid-cols-1 md:grid-cols-2 gap-6"
        >
          {/* Age */}
          <div>
            <label className="block font-medium mb-1">Age (years)</label>
            <input
              type="number"
              name="age"
              className="w-full rounded-lg border border-gray-300 px-4 py-2 focus:ring-2 focus:ring-red-500 focus:outline-none"
              placeholder="e.g. 55"
              required
              onChange={handleChange}
            />
          </div>

          {/* Sex */}
          <div>
            <label className="block font-medium mb-1">Gender</label>
            <select
              name="sex"
              className="w-full rounded-lg border border-gray-300 px-4 py-2 bg-white focus:ring-2 focus:ring-red-500"
              onChange={handleChange}
            >
              <option>Male</option>
              <option>Female</option>
            </select>
          </div>

          {/* Chest Pain */}
          <div>
            <label className="block font-medium mb-1">Chest Pain Type</label>
            <select
              name="chest_pain_type"
              className="w-full rounded-lg border border-gray-300 px-4 py-2"
              onChange={handleChange}
            >
              <option>Typical Angina</option>
              <option>Atypical Angina</option>
              <option>Non-anginal Pain</option>
              <option>Asymptomatic</option>
            </select>
          </div>

          {/* Resting BP */}
          <div>
            <label className="block font-medium mb-1">
              Resting Blood Pressure
            </label>
            <input
              type="number"
              name="resting_bp"
              className="w-full rounded-lg border border-gray-300 px-4 py-2"
              placeholder="e.g. 140 mmHg"
              required
              onChange={handleChange}
            />
          </div>

          {/* Cholesterol */}
          <div>
            <label className="block font-medium mb-1">Cholesterol</label>
            <input
              type="number"
              name="cholesterol"
              className="w-full rounded-lg border border-gray-300 px-4 py-2"
              placeholder="e.g. 240 mg/dl"
              required
              onChange={handleChange}
            />
          </div>

          {/* Fasting Blood Sugar */}
          <div>
            <label className="block font-medium mb-1">
              Fasting Blood Sugar
            </label>
            <select
              name="fasting_bs"
              className="w-full rounded-lg border border-gray-300 px-4 py-2"
              onChange={handleChange}
            >
              <option>{'<= 120 mg/dl'}</option>
              <option>{'> 120 mg/dl'}</option>
            </select>
          </div>

          {/* ECG */}
          <div>
            <label className="block font-medium mb-1">Resting ECG</label>
            <select
              name="resting_ecg"
              className="w-full rounded-lg border border-gray-300 px-4 py-2"
              onChange={handleChange}
            >
              <option>Normal</option>
              <option>ST-T wave Abnormality</option>
              <option>Left Ventricular Hypertrophy</option>
            </select>
          </div>

          {/* Max HR */}
          <div>
            <label className="block font-medium mb-1">
              Maximum Heart Rate
            </label>
            <input
              type="number"
              name="max_hr"
              className="w-full rounded-lg border border-gray-300 px-4 py-2"
              placeholder="e.g. 160"
              required
              onChange={handleChange}
            />
          </div>

          {/* Exercise Angina */}
          <div>
            <label className="block font-medium mb-1">
              Exercise Induced Angina
            </label>
            <select
              name="exercise_angina"
              className="w-full rounded-lg border border-gray-300 px-4 py-2"
              onChange={handleChange}
            >
              <option>No</option>
              <option>Yes</option>
            </select>
          </div>

          {/* Oldpeak */}
          <div>
            <label className="block font-medium mb-1">Oldpeak</label>
            <input
              type="number"
              step="0.1"
              name="oldpeak"
              className="w-full rounded-lg border border-gray-300 px-4 py-2"
              placeholder="e.g. 1.5"
              required
              onChange={handleChange}
            />
          </div>

          {/* ST Slope */}
          <div>
            <label className="block font-medium mb-1">ST Slope</label>
            <select
              name="st_slope"
              className="w-full rounded-lg border border-gray-300 px-4 py-2"
              onChange={handleChange}
            >
              <option>Upsloping</option>
              <option>Flat</option>
              <option>Downsloping</option>
            </select>
          </div>

          {/* Submit */}
          <button
            type="submit"
            className="md:col-span-2 mt-4 bg-red-600 hover:bg-red-700 text-white py-3 rounded-xl text-lg font-semibold transition"
            disabled={loading}
          >
            {loading ? "🔍 Analyzing..." : "❤️ Predict Heart Risk"}
          </button>
        </form>

        {/* Result */}
        {result && (
          <div className="mt-8 space-y-6">
            {/* Prediction Summary */}
            <div
              className={`border rounded-xl p-5 ${getRiskLevelColor(
                result.ann_prediction.result
              )}`}
            >
              <h2 className="text-xl font-bold mb-3">📊 Prediction Result</h2>
              <div className="space-y-2">
                <p>
                  <b>ANN Risk Level:</b> {result.ann_prediction.result}
                </p>
                <p>
                  <b>Confidence:</b> {result.ann_prediction.probability}%
                </p>
                <p>
                  <b>Random Forest:</b> {result.rf_prediction}
                </p>
                <p>
                  <b>Logistic Regression:</b> {result.lr_prediction}
                </p>
              </div>
            </div>

            {/* Risk Factors Section */}
            {result.risk_factors && result.risk_factors.length > 0 && (
              <div className="bg-blue-50 border border-blue-200 rounded-xl p-5">
                <h2 className="text-xl font-bold mb-4 text-blue-900">
                  📋 Why This Prediction?
                </h2>
                <p className="text-sm text-blue-800 mb-4">
                  These are the top contributing factors based on your clinical data:
                </p>
                <div className="space-y-4">
                  {result.risk_factors.map((factor, idx) => (
                    <div
                      key={idx}
                      className="bg-white rounded-lg p-4 border border-blue-100"
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <span className="text-2xl">
                            {getImpactIcon(factor.impact, factor.direction)}
                          </span>
                          <div>
                            <span className="font-semibold text-gray-800">
                              {factor.feature}
                            </span>
                            <span className="text-gray-600 ml-2">
                              ({factor.value})
                            </span>
                          </div>
                        </div>
                        <span
                          className={`text-sm font-medium capitalize ${getImpactColor(
                            factor.impact
                          )}`}
                        >
                          {factor.impact} Impact
                        </span>
                      </div>
                      <p className="text-sm text-gray-700 mb-2">
                        {factor.description}
                      </p>
                      <div className="flex items-center gap-2">
                        <div className="flex-1 bg-gray-200 rounded-full h-2 overflow-hidden">
                          <div
                            className={`h-full ${
                              factor.direction === "increases risk"
                                ? "bg-red-500"
                                : "bg-green-500"
                            }`}
                            style={{
                              width: `${Math.min(
                                factor.impact_score * 100,
                                100
                              )}%`,
                            }}
                          ></div>
                        </div>
                        <span className="text-xs text-gray-500">
                          {factor.direction}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Recommendations Section */}
            {result.recommendations && result.recommendations.length > 0 && (
              <div className="bg-purple-50 border border-purple-200 rounded-xl p-5">
                <h2 className="text-xl font-bold mb-4 text-purple-900">
                  💡 How to Lower Your Risk
                </h2>
                <p className="text-sm text-purple-800 mb-4">
                  Personalized recommendations based on your risk factors:
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {result.recommendations.map((rec, idx) => (
                    <div
                      key={idx}
                      className="bg-white rounded-lg p-4 border border-purple-100 flex flex-col"
                    >
                      <div className="flex items-start justify-between mb-2">
                        <span className="text-3xl">{rec.icon}</span>
                        <span
                          className={`text-xs px-2 py-1 rounded-full border font-medium ${getPriorityBadgeColor(
                            rec.priority
                          )}`}
                        >
                          {rec.priority.toUpperCase()}
                        </span>
                      </div>
                      <h3 className="font-semibold text-gray-800 mb-2">
                        {rec.title}
                      </h3>
                      <p className="text-sm text-gray-600 flex-1">
                        {rec.description}
                      </p>
                      <div className="mt-3 pt-3 border-t border-gray-100">
                        <span className="text-xs text-gray-500 capitalize">
                          {rec.category} Recommendation
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Medical Disclaimer */}
            <div className="bg-gray-100 border border-gray-300 rounded-lg p-4 text-sm text-gray-700">
              <p className="font-semibold mb-1">⚠️ Medical Disclaimer</p>
              <p>
                This prediction is generated by AI models and is for
                informational purposes only. It is not a substitute for
                professional medical advice, diagnosis, or treatment. Always
                consult with a qualified healthcare provider for medical
                concerns.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
