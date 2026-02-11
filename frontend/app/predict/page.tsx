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
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>,
  ) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setResult(null);
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
        return "bg-red-100 text-red-800 ";
      case "medium":
        return "bg-yellow-100 text-yellow-800";
      case "low":
        return "bg-green-100 text-green-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

  const handleExample = () => {
    setForm({
      age: "60",
      sex: "Male",
      chest_pain_type: "Non-anginal Pain",
      resting_bp: "100",
      cholesterol: "90",
      fasting_bs: "<= 120 mg/dl",
      resting_ecg: "Left Ventricular Hypertrophy",
      max_hr: "110",
      exercise_angina: "Yes",
      oldpeak: "1",
      st_slope: "Flat",
    });
  };

  return (
    <div className="min-h-screen flex justify-center items-start p-4 py-8">
      <div className="w-full max-w-6xl rounded-2xl shadow-xl p-6 md:p-10 bg-black/10 backdrop-blur-md border border-white/20">
        <h1 className="text-3xl font-bold text-center text-brand-fg mb-2">
          Heart Disease Prediction
        </h1>
        <p className="text-center text-brand-fg/80 mb-8">
          Please fill the form carefully. All fields are required.
        </p>

        <div>
          <button
            type="button"
            onClick={()=> handleExample()}
            className="mb-4 bg-brand-fg hover:bg-brand-fg/90 text-brand-bg py-2 px-4 rounded-lg text-sm transition"
          >
            Load Example Data
          </button>
        </div>

        <form
          onSubmit={handleSubmit}
          className="grid grid-cols-1 md:grid-cols-2 gap-6 text-brand-fg"
        >
          {/* Age */}
          <div>
            <label className="block font-medium mb-1">Age (years)</label>
            <input
              type="number"
              name="age"
              value={form.age}
              className="w-full rounded-lg border px-4 py-2 outline-none focus:ring-2 bg-brand-fg/10 border-brand-fg/30"
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
              className="w-full rounded-lg border px-4 py-2 outline-none focus:ring-2 bg-brand-fg/10 border-brand-fg/30"
              onChange={handleChange}
              value={form.sex}
            >
              <option className="bg-brand-fg text-brand-bg">Male</option>
              <option className="bg-brand-fg text-brand-bg">Female</option>
            </select>
          </div>

          {/* Chest Pain */}
          <div>
            <label className="block font-medium mb-1">Chest Pain Type</label>
            <select
              name="chest_pain_type"
              className="w-full rounded-lg border px-4 py-2 outline-none focus:ring-2 bg-brand-fg/10 border-brand-fg/30"
              onChange={handleChange}
              value={form.chest_pain_type}
            >
              <option className="bg-brand-fg text-brand-bg">
                Typical Angina
              </option>
              <option className="bg-brand-fg text-brand-bg">
                Atypical Angina
              </option>
              <option className="bg-brand-fg text-brand-bg">
                Non-anginal Pain
              </option>
              <option className="bg-brand-fg text-brand-bg">
                Asymptomatic
              </option>
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
              className="w-full rounded-lg border px-4 py-2 outline-none focus:ring-2 bg-brand-fg/10 border-brand-fg/30"
              placeholder="e.g. 140 mmHg"
              required
              onChange={handleChange}
              value={form.resting_bp}
            />
          </div>

          {/* Cholesterol */}
          <div>
            <label className="block font-medium mb-1">Cholesterol</label>
            <input
              type="number"
              name="cholesterol"
              className="w-full rounded-lg border px-4 py-2 outline-none focus:ring-2 bg-brand-fg/10 border-brand-fg/30"
              placeholder="e.g. 240 mg/dl"
              required
              onChange={handleChange}
              value={form.cholesterol}
            />
          </div>

          {/* Fasting Blood Sugar */}
          <div>
            <label className="block font-medium mb-1">
              Fasting Blood Sugar
            </label>
            <select
              name="fasting_bs"
              className="w-full rounded-lg border px-4 py-2 outline-none focus:ring-2 bg-brand-fg/10 border-brand-fg/30"
              onChange={handleChange}
              value={form.fasting_bs}
            >
              <option className="bg-brand-fg text-brand-bg">
                {"<= 120 mg/dl"}
              </option>
              <option className="bg-brand-fg text-brand-bg">
                {"> 120 mg/dl"}
              </option>
            </select>
          </div>

          {/* ECG */}
          <div>
            <label className="block font-medium mb-1">Resting ECG</label>
            <select
              name="resting_ecg"
              className="w-full rounded-lg border px-4 py-2 outline-none focus:ring-2 bg-brand-fg/10 border-brand-fg/30"
              onChange={handleChange}
              value={form.resting_ecg}
            >
              <option className="bg-brand-fg text-brand-bg">Normal</option>
              <option className="bg-brand-fg text-brand-bg">
                ST-T wave Abnormality
              </option>
              <option className="bg-brand-fg text-brand-bg">
                Left Ventricular Hypertrophy
              </option>
            </select>
          </div>

          {/* Max HR */}
          <div>
            <label className="block font-medium mb-1">Maximum Heart Rate</label>
            <input
              type="number"
              name="max_hr"
              className="w-full rounded-lg border px-4 py-2 outline-none focus:ring-2 bg-brand-fg/10 border-brand-fg/30"
              placeholder="e.g. 160"
              required
              onChange={handleChange}
              value={form.max_hr}
            />
          </div>

          {/* Exercise Angina */}
          <div>
            <label className="block font-medium mb-1">
              Exercise Induced Angina
            </label>
            <select
              name="exercise_angina"
              className="w-full rounded-lg border px-4 py-2 outline-none focus:ring-2 bg-brand-fg/10 border-brand-fg/30"
              onChange={handleChange}
              value={form.exercise_angina}
            >
              <option className="bg-brand-fg text-brand-bg">No</option>
              <option className="bg-brand-fg text-brand-bg">Yes</option>
            </select>
          </div>

          {/* Oldpeak */}
          <div>
            <label className="block font-medium mb-1">Oldpeak</label>
            <input
              type="number"
              step="0.1"
              name="oldpeak"
              className="w-full rounded-lg border px-4 py-2 outline-none focus:ring-2 bg-brand-fg/10 border-brand-fg/30"
              placeholder="e.g. 1.5"
              required
              onChange={handleChange}
              value={form.oldpeak}
            />
          </div>

          {/* ST Slope */}
          <div>
            <label className="block font-medium mb-1">ST Slope</label>
            <select
              name="st_slope"
              className="w-full rounded-lg border px-4 py-2 outline-none focus:ring-2 bg-brand-fg/10 border-brand-fg/30"
              onChange={handleChange}
              value={form.st_slope}
            >
              <option className="bg-brand-fg text-brand-bg">Upsloping</option>
              <option className="bg-brand-fg text-brand-bg">Flat</option>
              <option className="bg-brand-fg text-brand-bg">Downsloping</option>
            </select>
          </div>

          {/* Submit */}
          <button
            type="submit"
            className="md:col-span-2 mt-4 bg-brand-fg hover:bg-brand-fg/90 text-brand-bg py-3 rounded-xl text-lg font-semibold transition"
            disabled={loading}
          >
            {loading ? "🔍 Analyzing..." : "❤️ Predict Heart Risk"}
          </button>
        </form>

        {/* Result */}
        {result && (
          <div className="mt-8 space-y-6">
            {/* Prediction Summary */}
            <div className={`rounded-xl p-5 bg-brand-fg/10 border-brand-fg/30 text-brand-fg`}>
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
              <div className="bg-brand-fg/10 text-brand-fg rounded-xl p-5">
                <h2 className="text-xl font-bold mb-4">
                  📋 Why This Prediction?
                </h2>
                <p className="text-sm mb-4">
                  These are the top contributing factors based on your clinical
                  data:
                </p>
                <div className="space-y-4 text-brand-fg">
                  {result.risk_factors.map((factor, idx) => (
                    <div
                      key={idx}
                      className="bg-brand-fg/10 border-brand-fg/30 rounded-lg p-4 border"
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <span className="text-2xl">
                            {getImpactIcon(factor.impact, factor.direction)}
                          </span>
                          <div>
                            <span className="font-semibold">
                              {factor.feature}
                            </span>
                            <span className=" ml-2">
                              ({factor.value})
                            </span>
                          </div>
                        </div>
                        <span
                          className={`text-sm font-medium capitalize ${getImpactColor(
                            factor.impact,
                          )}`}
                        >
                          {factor.impact} Impact
                        </span>
                      </div>
                      <p className="text-sm mb-2">
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
                                100,
                              )}%`,
                            }}
                          ></div>
                        </div>
                        <span className="text-xs text-brand-fg/60">
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
              <div className="bg-brand-fg/10 rounded-xl p-5 text-brand-fg">
                <h2 className="text-xl font-bold mb-4">
                  💡 How to Lower Your Risk
                </h2>
                <p className="text-sm mb-4">
                  Personalized recommendations based on your risk factors:
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {result.recommendations.map((rec, idx) => (
                    <div
                      key={idx}
                      className="bg-brand-fg/10 border-brand-fg/30 rounded-lg p-4 border flex flex-col"
                    >
                      <div className="flex items-start justify-between mb-2">
                        <span className="text-3xl">{rec.icon}</span>
                        <span
                          className={`text-xs px-2 py-1 rounded-full font-medium ${getPriorityBadgeColor(
                            rec.priority,
                          )}`}
                        >
                          {rec.priority.toUpperCase()}
                        </span>
                      </div>
                      <h3 className="font-semibold mb-2">
                        {rec.title}
                      </h3>
                      <p className="text-sm flex-1 text-brand-fg/80">
                        {rec.description}
                      </p>
                      <div className="mt-3 pt-3 border-t">
                        <span className="text-xs capitalize text-brand-fg/60">
                          {rec.category} Recommendation
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Medical Disclaimer */}
            <div className="bg-brand-fg/10 border-brand-fg/30 rounded-lg p-4 text-sm text-brand-fg/80 border">
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
