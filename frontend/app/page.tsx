import Link from "next/link";
import { JSX } from "react";

export default function Home(): JSX.Element {
  return (
    <div className="text-center">
      <section className="py-20">
        <h1 className="text-4xl md:text-5xl font-bold mb-4">
          AI-Based Heart Disease Prediction
        </h1>

        <p className="text-gray-600 max-w-2xl mx-auto mb-8">
          Predict heart disease risk using clinical data and compare multiple
          machine learning models including ANN, Random Forest, and Logistic
          Regression.
        </p>

        <Link
          href="/predict"
          className="inline-block bg-blue-600 text-white px-8 py-3 rounded-lg text-lg hover:bg-blue-700 transition"
        >
          Start Prediction →
        </Link>
      </section>

      <section className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-10">
        <Feature
          title="🧠 AI Prediction"
          text="Uses trained ML models to classify heart disease risk accurately."
        />
        <Feature
          title="📊 Model Comparison"
          text="Compare ANN, Random Forest, and Logistic Regression accuracy."
        />
        <Feature
          title="🗂 Patient History"
          text="Securely store and review previous predictions."
        />
      </section>
    </div>
  );
}

function Feature({
  title,
  text,
}: {
  title: string;
  text: string;
}): JSX.Element {
  return (
    <div className="bg-white p-6 rounded shadow hover:shadow-lg transition">
      <h3 className="text-xl font-semibold mb-2">{title}</h3>
      <p className="text-gray-600">{text}</p>
    </div>
  );
}
